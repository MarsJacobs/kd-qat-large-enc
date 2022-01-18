from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
import pickle
import copy

import math

import numpy as np
import numpy
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
from torch.utils.tensorboard import SummaryWriter

from torch.nn import CrossEntropyLoss, MSELoss
from torchmetrics import HammingDistance

from transformer import BertForSequenceClassification,WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling_quant import BertForSequenceClassification as QuantBertForSequenceClassification
from transformer import BertTokenizer
from transformer import BertAdam
from transformer import BertConfig
from utils_glue import *

import matplotlib.pyplot as plt
import torch.nn.functional as F

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)


    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids

def do_logging(run, student_model, teacher_model, new_eval_dataloader, device, global_step):
    
    nb_steps = 0

    kl_div_sum = [0 for i in range(12)]
    st_sep_avg_sum = [0 for i in range(12)]; st_cls_avg_sum = [0 for i in range(12)]; tc_sep_avg_sum = [0 for i in range(12)]; tc_cls_avg_sum = [0 for i in range(12)]
    cover_sum = [0 for i in range(12)]
    
    for _, batch_ in enumerate(new_eval_dataloader):
        batch_ = tuple(t.to(device) for t in batch_)

        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_id, seq_length = batch_

            teacher_logits, teacher_atts, teacher_reps, teacher_probs, teacher_values = teacher_model(input_ids, segment_ids, input_mask)
            student_logits, student_atts, student_reps, student_probs, student_values = student_model(input_ids, segment_ids, input_mask, teacher_probs=teacher_probs)
            
            for i, (student_prob, teacher_prob) in enumerate(zip(student_probs, teacher_probs)): 
                # Loss_ranking_sum = 0
                # hamming_sum = 0
                # head
                for head in range(12):

                    # Attention Map
                    student_attn_map = student_prob[0][head][:seq_length,:seq_length]
                    teacher_attn_map = teacher_prob[0][head][:seq_length,:seq_length]

                    # KL Divergence
                    kl_div = F.kl_div(student_attn_map.log(), teacher_attn_map, reduction='batchmean')
                    kl_div_sum[i] += kl_div

                    # Special Token Prob Mean
                    st_sep_avg = student_attn_map[:,-1].mean()
                    st_cls_avg = student_attn_map[:,0].mean()
                    st_sep_avg_sum[i] += st_sep_avg
                    st_cls_avg_sum[i] += st_cls_avg
                    
                    # Ground Truth
                    tc_sep_avg = teacher_attn_map[:,-1].mean()
                    tc_cls_avg = teacher_attn_map[:,0].mean()
                    tc_sep_avg_sum[i] += tc_sep_avg
                    tc_cls_avg_sum[i] += tc_cls_avg

                    # Coverage Test
                    coverage_head_sum = 0
                    for k in range(student_attn_map.shape[0]):
                        st_argsort = student_attn_map[k].sort(descending=True)[1]
                        tc_argsort = teacher_attn_map[k].sort(descending=True)[1][:5] # Top-5
                        
                        max_idx = 0
                        for idx in tc_argsort: # Teacher Top-5                             
                            tmp = torch.where(st_argsort == idx)
                            max_idx = max(tmp[0].item(), max_idx)
                        
                        coverage_ratio = max_idx / student_attn_map.shape[0]
                        coverage_head_sum += coverage_ratio
                    
                    coverage_head = coverage_head_sum / student_attn_map.shape[0]
                    cover_sum[i] += coverage_head
                    
                    nb_steps += 1

    nb_steps = nb_steps / 12
    
    for l in range(12):
        run[f"attn/L{l}_KLdiv_mean"].log(value=kl_div_sum[l] / nb_steps, step=global_step)
        run[f"attn/L{l}_st_SepProb_mean"].log(value=st_sep_avg_sum[l] / nb_steps, step=global_step)
        run[f"attn/L{l}_st_ClsProb_mean"].log(value=st_cls_avg_sum[l] / nb_steps, step=global_step)
        run[f"attn/L{l}_tc_SepProb_mean"].log(value=tc_sep_avg_sum[l] / nb_steps, step=global_step)
        run[f"attn/L{l}_tc_ClsProb_mean"].log(value=tc_cls_avg_sum[l] / nb_steps, step=global_step)
        run[f"attn/L{l}_cover_mean"].log(value=cover_sum[l] / nb_steps, step=global_step)
                            


def do_eval(model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels, teacher_model=None):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for _,batch_ in enumerate(eval_dataloader):
        batch_ = tuple(t.to(device) for t in batch_)
        
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

            # teacher attnmap test
            if teacher_model is not None:
                logits, _, _, teacher_probs, _ = teacher_model(input_ids, segment_ids, input_mask)
                logits, _, _, _, _ = model(input_ids, segment_ids, input_mask, teacher_probs=teacher_probs)
            else:
                logits, _, _, _, _ = model(input_ids, segment_ids, input_mask)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss
    return result

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return torch.sum((- targets_prob * student_likelihood), dim=-1).mean()

def main():

    # ================================================================================  #
    # ArgParse
    # ================================================================================ #
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default='data',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir",
                        default='models',
                        type=str,
                        help="The model dir.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The models directory.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        help="The models directory.")
    parser.add_argument("--task_name",
                        default='sst-2',
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='output',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    # MSKIM Add GT Loss Option
    parser.add_argument('--gt_loss',
                        action='store_true',
                        help="Whether to use GT Label Loss")

    parser.add_argument('--pred_distill',
                        action='store_true',
                        help="Whether to distil with task layer")
    parser.add_argument('--intermediate_distill',
                        action='store_true',
                        help="Whether to distil with intermediate layers")
    parser.add_argument('--save_fp_model',
                        action='store_true',
                        help="Whether to save fp32 model")
                        
    parser.add_argument('--save_quantized_model',
                        default=False, type=str2bool,
                        help="Whether to save quantized model")

    parser.add_argument("--weight_bits",
                        default=2,
                        type=int,
                        choices=[2,8],
                        help="Quantization bits for weight.")
    parser.add_argument("--input_bits",
                        default=8,
                        type=int,
                        help="Quantization bits for activation.")
    parser.add_argument("--gpus",
                        default=1,
                        type=int,
                        help="Number of GPUs to use")
    parser.add_argument("--clip_val",
                        default=2.5,
                        type=float,
                        help="Initial clip value.")

    parser.add_argument('--act_quant',
                        default=True, type=str2bool,
                        help="Whether to quantize activation")
    
    parser.add_argument('--neptune',
                        default=True, type=str2bool,
                        help="neptune logging option")
    
    #MSKIM Quantization Option
    parser.add_argument("--quantizer",
                        default="ternary",
                        type=str,
                        help="Quantization Method")

    #MSKIM Quantization Range Option
    parser.add_argument('--quantize',
                        default =False, type=str2bool,
                        help="Whether to quantize student model")

    parser.add_argument('--ffn_1',
                        default =False, type=str2bool,
                        help="Whether to quantize Feed Forward Network")
    
    parser.add_argument('--ffn_2',
                        default =False, type=str2bool,
                        help="Whether to quantize Feed Forward Network")
    
    parser.add_argument('--qkv',
                        default =False, type=str2bool,
                        help="Whether to quantize Query, Key, Value Mapping Weight Matrix")
    
    parser.add_argument('--emb',
                        default =False, type=str2bool,
                        help="Whether to quantize Embedding Layer")

    parser.add_argument('--cls',
                        default =False, type=str2bool,
                        help="Whether to quantize Classifier Dense Layer")
    
    parser.add_argument('--aug_train',
                        default =False, type=str2bool,
                        help="Whether to use augmented data or not")

    parser.add_argument('--clipping',
                        default =False, type=str2bool,
                        help="Whether to use FP Weight Clipping")
    
    parser.add_argument('--downstream',
                        default =False, type=str2bool,
                        help="Downstream mode")
    
    parser.add_argument('--prob_log',
                        default =False, type=str2bool,
                        help="attention prob logging option")

    parser.add_argument('--attn_test',
                        default =False, type=str2bool,
                        help="attention prob logging option")
                        
    parser.add_argument('--gradient_scaling',
                        default =1, type=float,
                        help="LSQ gradient scaling")

    parser.add_argument("--layer_num",
                        default=-1,
                        type=int,
                        help="Number of layer to quantize (-1 : Quantize every layer")
    
    parser.add_argument("--kd_layer_num",
                        default=-1,
                        type=int,
                        help="Number of layer to Apply KD (-1 : Distill every layer")
    
    parser.add_argument("--mean_scale",
                        default=0.7,
                        type=float,
                        help="Ternary Clipping Value Scale Value")
    
    parser.add_argument("--clip_ratio",
                        default=0.7,
                        type=float,
                        help="Clip Value Init raio")

    parser.add_argument("--clip_method",
                        default="minmax",
                        type=str,
                        help="Clip Value Init Method")

    parser.add_argument("--exp_name",
                        default="",
                        type=str,
                        help="Output Directory Name")
    
    parser.add_argument("--init_scaling",
                        default=1.,
                        type=float,
                        help="LSQ/PACT Clipping Init Value Scaling Value")
    
    parser.add_argument("--lr_scaling",
                        default=1.,
                        type=float,
                        help="LSQ/PACT Clipping Learning Rate Scaling Value")
    
    parser.add_argument("--clip_wd",
                        default=0.3,
                        type=float,
                        help="PACT Clip Value Weight Decay")

    parser.add_argument("--attnmap_coeff",
                        default=1,
                        type=float,
                        help="attnmap loss coeff")
    
    parser.add_argument("--cls_coeff",
                        default=1,
                        type=float,
                        help="cls loss coeff")
    
    parser.add_argument("--att_coeff",
                        default=1,
                        type=float,
                        help="att loss coeff")

    parser.add_argument("--rep_coeff",
                        default=1,
                        type=float,
                        help="rep loss coeff")

    parser.add_argument("--aug_N",
                        default=30,
                        type=int,
                        help="Data Augmentation N Number")
    
    parser.add_argument('--clip_teacher',
                        default=False, type=str2bool,
                        help="use FP weight clipped teacher model")

    parser.add_argument('--attn_distill',
                        default =True, type=str2bool,
                        help="attention Score Distill Option")

    parser.add_argument('--rep_distill',
                        default =True, type=str2bool,
                        help="Transformer Layer output Distill Option")

    parser.add_argument('--attnmap_distill',
                        default =True, type=str2bool,
                        help="attention Map Distill Option")
    
    parser.add_argument('--value_relation',
                        default =False, type=str2bool,
                        help="attention Map Distill Option")
    parser.add_argument('--teacher_attnmap',
                        default =False, type=str2bool,
                        help="attention Map Distill Option")
    args = parser.parse_args() 
    
    # ================================================================================  #
    # Logging setup
    # ================================================================================ #
    run = None
    if args.neptune:
        import neptune.new as neptune
        run = neptune.init(project='niceball0827/' + args.task_name.upper(),
                    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLC\
                    JhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YjM\
                    0ZTYwMi1kNjQwLTQ4NGYtOTYxMy03Mjc5ZmVkMzY2YTgifQ==')

    # ================================================================================  #
    # Load Directory
    # ================================================================================ #
    tb_dir = os.path.join(args.output_dir, "Tensorboard", args.task_name)
    if not os.path.exists(tb_dir):
        os.mkdir(tb_dir)
    #summaryWriter = SummaryWriter(tb_dir)

    logger.info('The args: {}'.format(args))
    task_name = args.task_name.lower()
    data_dir = os.path.join(args.data_dir,task_name.upper())
    output_dir = os.path.join(args.output_dir,task_name)
    processed_data_dir = os.path.join(data_dir,'preprocessed')
    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
# ================================================================================  #
# Load Pths
# ================================================================================ #

    if args.student_model is None:
        if not args.downstream:
            #args.student_model = os.path.join("output", "BERT_base",task_name.upper())
            args.student_model = os.path.join("models",task_name.upper())
            #args.student_model = os.path.join("models", "BERT_base")
            #args.student_model = os.path.join(args.model_dir, "FFN")
        else:
            args.student_model = os.path.join("models", "BERT_base")
        #args.student_model = os.path.join(args.model_dir, "FFN")
    if args.teacher_model is None:
        if args.clip_teacher :
            args.teacher_model = os.path.join("output", "cola", "quant", "0103")
            #args.teacher_model = os.path.join(args.model_dir, "FFN")
        else:
            #args.teacher_model = os.path.join("output", "BERT_base",task_name.upper())
            args.teacher_model = os.path.join("models",task_name.upper())
        
        

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification"
    }

    default_params = {
        "cola": {"max_seq_length": 64,"batch_size":16,"eval_step": 50 if args.aug_train else 50}, # No Aug : 50 Aug : 400
        "mnli": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "mrpc": {"max_seq_length": 128,"batch_size":32,"eval_step":100},
        "sst-2": {"max_seq_length": 64,"batch_size":32,"eval_step":200},
        "sts-b": {"max_seq_length": 128,"batch_size":32,"eval_step":50},
        "qqp": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "qnli": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "rte": {"max_seq_length": 128,"batch_size":32,"eval_step":570 if args.aug_train else 100}
    }

    acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte"]
    corr_tasks = ["sts-b"]
    mcc_tasks = ["cola"]

    # ================================================================================  #
    # prepare devices
    # ================================================================================ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = args.gpus
    

    # ================================================================================  #
    # prepare seed
    # ================================================================================ #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
 
    if task_name in default_params:
        args.batch_size = default_params[task_name]["batch_size"]
        if n_gpu > 0:
            args.batch_size = int(args.batch_size*n_gpu)
        args.max_seq_length = default_params[task_name]["max_seq_length"]
        args.eval_step = default_params[task_name]["eval_step"]
    
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # ================================================================================  #
    # Load Vocab FIle -> Tokenization 
    # ================================================================================ #
    tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=True)
    vocab = torch.load("vocab_dict.pth")

    # ================================================================================  #
    # Dataset Setup
    # ================================================================================ #
    if args.aug_train:
        try:
            train_file = os.path.join(processed_data_dir,f'aug_data_{args.aug_N}.pkl')
            train_features = pickle.load(open(train_file,'rb'))
        except:
            train_examples = processor.get_aug_examples(data_dir, args.aug_N)
            train_features = convert_examples_to_features(train_examples, label_list,
                                            args.max_seq_length, tokenizer, output_mode)
            with open(train_file, 'wb') as f:
                pickle.dump(train_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            train_file = os.path.join(processed_data_dir,'data.pkl')
            train_features = pickle.load(open(train_file,'rb'))
            
        except:
            train_examples = processor.get_train_examples(data_dir)
            train_features = convert_examples_to_features(train_examples, label_list,
                                            args.max_seq_length, tokenizer, output_mode)
            with open(train_file, 'wb') as f:
                pickle.dump(train_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    num_train_epochs = args.num_train_epochs if not args.aug_train else 1
    num_train_optimization_steps = math.ceil(len(train_features) / args.batch_size) * num_train_epochs
    
    train_data, _ = get_tensor_data(output_mode, train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    
    try:
        dev_file = train_file = os.path.join(processed_data_dir,'dev.pkl')
        eval_features = pickle.load(open(dev_file,'rb'))
    except:
        eval_examples = processor.get_dev_examples(data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        with open(dev_file, 'wb') as f:
                pickle.dump(eval_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    
    # Eval Data Extraction for logging data (10%)
    new_eval_features = eval_features[:int(len(eval_features)*0.1)]
    new_eval_data, new_eval_labels = get_tensor_data(output_mode, new_eval_features)
    new_eval_sampler = SequentialSampler(new_eval_data)
    new_eval_dataloader = DataLoader(eval_data, sampler=new_eval_sampler, batch_size=1)
    logger.info("  Num examples for Logging = %d", len(new_eval_features))

    if task_name == "mnli":
        processor = processors["mnli-mm"]()
        try:
            dev_mm_file = train_file = os.path.join(processed_data_dir,'dev-mm_data.pkl')
            mm_eval_features = pickle.load(open(dev_mm_file,'rb'))
        except:
            mm_eval_examples = processor.get_dev_examples(data_dir)
            mm_eval_features = convert_examples_to_features(
                mm_eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
            with open(dev_mm_file, 'wb') as f:
                pickle.dump(mm_eval_features, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        mm_eval_data, mm_eval_labels = get_tensor_data(output_mode, mm_eval_features)
        logger.info("  Num examples = %d", len(mm_eval_features))

        mm_eval_sampler = SequentialSampler(mm_eval_data)
        mm_eval_dataloader = DataLoader(mm_eval_data, sampler=mm_eval_sampler,
                                        batch_size=args.batch_size)


    # ================================================================================  #
    # Build Teacher Model
    # ================================================================================ # 
    
    
    # Clipped Teacher
    if args.clip_teacher:
        teacher_config = BertConfig.from_pretrained(args.teacher_model)
        teacher_model = QuantBertForSequenceClassification.from_pretrained(args.teacher_model, config = teacher_config, num_labels=num_labels)
    else:
        teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_model)

    teacher_model.to(device)
    teacher_model.eval()
    if n_gpu > 1:
        teacher_model = torch.nn.DataParallel(teacher_model, device_ids=range(n_gpu))
    
    result = do_eval(teacher_model, task_name, eval_dataloader,
                    device, output_mode, eval_labels, num_labels)
    logger.info(result)
    
    # ================================================================================  #
    # Save Teacher Model Peroformance for KD Training
    # ================================================================================ #
    if task_name in acc_tasks:
        if task_name in ['sst-2','mnli','qnli','rte']:
            fp32_performance = f"acc:{result['acc']}"
        elif task_name in ['mrpc','qqp']:
            fp32_performance = f"f1/acc:{result['f1']}/{result['acc']}"
    if task_name in corr_tasks:
        fp32_performance = f"pearson/spearmanr:{result['pearson']}/{result['spearmanr']}"

    if task_name in mcc_tasks:
        fp32_performance = f"mcc:{result['mcc']}"

    if task_name == "mnli":
        result = do_eval(teacher_model, 'mnli-mm', mm_eval_dataloader,
                            device, output_mode, mm_eval_labels, num_labels)
        fp32_performance += f"  mm-acc:{result['acc']}"
    fp32_performance = task_name +' fp32   ' + fp32_performance
    
    # ================================================================================  #
    # Build Student Model
    # ================================================================================ #
    student_config = BertConfig.from_pretrained(args.student_model, 
                                                quantize_act=args.act_quant,
                                                weight_bits = args.weight_bits,
                                                input_bits = args.input_bits,
                                                clip_val = args.clip_val,
                                                quantize = args.quantize,
                                                ffn_q_1 = args.ffn_1,
                                                ffn_q_2 = args.ffn_2,
                                                qkv_q = args.qkv,
                                                emb_q = args.emb,
                                                cls_q = args.cls,
                                                clipping = args.clipping,
                                                layer_num = args.layer_num,
                                                mean_scale = args.mean_scale,
                                                quantizer = args.quantizer,
                                                init_scaling = args.init_scaling,
                                                clip_ratio = args.clip_ratio,
                                                gradient_scaling = args.gradient_scaling,
                                                clip_method = args.clip_method,
                                                teacher_attnmap = args.teacher_attnmap
                                                )
    
    student_model = QuantBertForSequenceClassification.from_pretrained(args.student_model, config = student_config, num_labels=num_labels)
    
    for name, module in student_model.named_modules():
        if hasattr(module,'weight_quantizer'):
            module.clip_initialize()
            #print(f"{name[13:]} {(module.weight.std()*3 / module.weight.max()).item():.2f} {(module.weight.std()*sent_i / module.weight.min()).item():.2f}")
            
    student_model.to(device)
    
    # ================================================================================  #
    # Training Setting
    # ================================================================================ #
    
    if n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)
        
    # Prepare optimizer
    param_optimizer = list(student_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    clip_decay = ['clip_val', 'clip_valn']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in (no_decay+clip_decay))], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(cv in n for cv in clip_decay)],\
         'weight_decay': args.clip_wd if args.quantizer == "pact" else 0, 'lr': args.learning_rate * args.lr_scaling}
    ]
    
    schedule = 'warmup_linear'
    optimizer = BertAdam(optimizer_grouped_parameters,
                            schedule=schedule,
                            lr=args.learning_rate,
                            warmup=0.1,
                            t_total=num_train_optimization_steps)
    
    loss_mse = MSELoss()
    loss_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    hamming_distance = HammingDistance()
    
    global_step = 0
    best_dev_acc = 0.0
    previous_best = None
    
    tr_loss = 0.
    tr_att_loss = 0.
    tr_rep_loss = 0.
    tr_cls_loss = 0.

    
    # ================================================================================  #
    # Training Start
    # ================================================================================ #

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    # Init Clip Value Logging
    for n, p in student_model.named_parameters():
        if "clip_val" in n:
            run[f"clip_val/{n}"].log(p)
    
    
    for epoch_ in range(int(num_train_epochs)):
        logger.info("****************************** %d Epoch ******************************", epoch_)
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(train_dataloader):
            
            student_model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
            att_loss = 0.
            # MSKIM Added
            attmap_loss = 0.
            #vr_loss = 0

            rep_loss = 0.
            cls_loss = 0.
            
            loss = 0.
            
            with torch.no_grad():
                teacher_logits, teacher_atts, teacher_reps, teacher_probs, teacher_values = teacher_model(input_ids, segment_ids, input_mask)
            
            student_logits, student_atts, student_reps, student_probs, student_values = student_model(input_ids, segment_ids, input_mask, teacher_probs=teacher_probs)

            if args.gt_loss:
                lprobs = torch.nn.functional.log_softmax(student_logits, dim=-1)
                loss = torch.nn.functional.nll_loss(lprobs, label_ids, reduction='sum')

            if args.pred_distill:
                if output_mode == "classification":
                    cls_loss = soft_cross_entropy(student_logits,teacher_logits)
                elif output_mode == "regression":
                    cls_loss = loss_mse(student_logits, teacher_logits)
            
                cls_loss = cls_loss * args.cls_coeff
                l_cls_loss = cls_loss.item()
                

            # if args.value_relation:
            #     for i, (student_value, teacher_value) in enumerate(zip(student_values, teacher_values)):
                    
            #         kl_loss_vr = 0
                    
            #         for head in range(student_value.shape[1]):
            #             for word in range(student_value.shape[-1]):
            #                 input_st = F.log_softmax(student_value[:,head,:,:], dim=-1)
            #                 input_tc = F.softmax(teacher_value[:,head,:,:], dim=-1)
            #                 kl_loss_vr += torch.nn.KLDivLoss(reduction='batchmean')(input_st, input_tc)
                    
            #         vr_loss += kl_loss_vr

            if args.intermediate_distill:
                if args.attnmap_distill or args.attn_distill:
                                        
                    # for i, (student_prob, teacher_prob) in enumerate(zip(student_probs, teacher_probs)):
                        
                    #     pearson_loss_tmp = 0
                        
                    #     for sent in range(teacher_prob.shape[0]):
                    #         for head in range(teacher_prob.shape[1]):
                    #             input_st = student_prob[sent, head, :seq_lengths[sent], :seq_lengths[sent]]
                    #             input_tc = teacher_prob [sent, head, :seq_lengths[sent], :seq_lengths[sent]]

                    #             #cos_loss_tmp += loss_cos(input_st, input_tc).mean(dim=0)
                                
                    #             pearson_loss_tmp += loss_cos(input_st - input_st.mean(dim=1), input_tc - input_tc.mean(dim=1)).sum()

                                
                        
                    #     attmap_loss += pearson_loss_tmp
                    


                    for i, (student_att, teacher_att) in enumerate(zip(student_atts, teacher_atts)):
                        
                        if args.attnmap_distill:
                            kl_loss_tmp = 0

                            for sent in range(teacher_att.shape[0]):
                                for head in range(teacher_att.shape[1]):
                                    # input_st = F.log_softmax(student_att[sent, head,:seq_lengths[sent],:seq_lengths[sent]], dim=-1)
                                    # input_tc = F.softmax(teacher_att[sent, head,:seq_lengths[sent],:seq_lengths[sent]], dim=-1)
                                    
                                    # loss_1 = torch.nn.KLDivLoss(reduction='batchmean')(input_st, input_tc)

                                    # input_st = F.softmax(student_att[sent, head,:seq_lengths[sent],:seq_lengths[sent]], dim=-1)
                                    # input_tc = F.log_softmax(teacher_att[sent, head,:seq_lengths[sent],:seq_lengths[sent]], dim=-1)
                                    
                                    # loss_2 = torch.nn.KLDivLoss(reduction='batchmean')(input_tc, input_st)
                                    # kl_loss_tmp += (loss_1 + loss_2) / 2
                                    input_st = F.log_softmax(student_att[sent, head,:seq_lengths[sent],:seq_lengths[sent]], dim=-1)
                                    input_tc = F.softmax(teacher_att[sent, head,:seq_lengths[sent],:seq_lengths[sent]], dim=-1)
                                    kl_loss_tmp += torch.nn.KLDivLoss(reduction='batchmean')(input_st, input_tc)

                            attmap_loss += kl_loss_tmp #* (i+1 / 12)
                            
                        
                            
                        if args.attn_distill:
                            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                                        student_att)
                            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                                        teacher_att)
                            if args.kd_layer_num != -1:
                                if i == args.kd_layer_num:
                                    tmp_loss = loss_mse(student_att, teacher_att)
                                else:
                                    tmp_loss = loss_mse(student_att, teacher_att)
                                    tmp_loss = tmp_loss * 0
                            else:
                                tmp_loss = loss_mse(student_att, teacher_att)

                            att_loss += tmp_loss
                    
                    attmap_loss = args.attnmap_coeff*attmap_loss
                    att_loss = args.att_coeff * att_loss

                    l_attmap_loss = attmap_loss.item()
                    l_att_loss = att_loss.item()

                if args.rep_distill:
                    for i, (student_rep, teacher_rep) in enumerate(zip(student_reps, teacher_reps)):
                        
                        if args.kd_layer_num != -1:
                            if i == args.kd_layer_num:
                                tmp_loss = loss_mse(student_rep, teacher_rep)
                            else:
                                tmp_loss = loss_mse(student_rep, teacher_rep)
                                tmp_loss = tmp_loss * 0
                        else:
                            tmp_loss = loss_mse(student_rep, teacher_rep)

                        rep_loss += tmp_loss
                
                rep_loss = args.rep_coeff * rep_loss
                l_rep_loss = rep_loss.item()
            


            loss += cls_loss + rep_loss + att_loss + attmap_loss 
            l_loss = loss.item()
            
            if n_gpu > 1:
                loss = loss.mean()

            

            loss.backward()        
            optimizer.step() 
            optimizer.zero_grad()
            
            global_step += 1
            # ================================================================================  #
            #  Attn Test
            # ================================================================================ #
            sent_i = None

            if global_step % args.eval_step == 0 or global_step == num_train_optimization_steps-1 or step == 0:
                logger.info("***** Running evaluation *****")
                
                logger.info("  {} step of {} steps".format(global_step, num_train_optimization_steps))
                
                if previous_best is not None:
                    logger.info(f"{fp32_performance}\nPrevious best = {previous_best}")

                student_model.eval()
                
                result = do_eval(student_model, task_name, eval_dataloader,
                                    device, output_mode, eval_labels, num_labels, teacher_model=teacher_model)
            
                result['global_step'] = global_step
                result['cls_loss'] = l_cls_loss
                result['att_loss'] = l_att_loss
                result['rep_loss'] = l_rep_loss
                result['loss'] = l_loss
                

                # Logging
                if run is not None:
                    
                    run["loss/total_loss"].log(value=l_loss, step=global_step)
                    run["loss/att_loss_loss"].log(value=l_att_loss, step=global_step)
                    run["loss/rep_loss_loss"].log(value=l_rep_loss, step=global_step)
                    run["loss/cls_loss_loss"].log(value=l_cls_loss, step=global_step)
                    run["loss/attmap_loss_loss"].log(value=l_attmap_loss, step=global_step)
                    run["metrics/lr"].log(value=optimizer.get_lr()[0], step=global_step)

                    if args.prob_log:
                        # logging metrics (attention device, special token probs..)
                        logger.info(f"  {global_step} step logging begins..")
                        do_logging(run, student_model, teacher_model, new_eval_dataloader, device, global_step)
                        logger.info(f"  {global_step} step logging done..")
                        # Hamming Distance
                        # st_sort = student_prob_plt.sort(descending=True, dim=-1)[1].detach().clone().cpu()
                        # tc_sort = teacher_prob_plt.sort(descending=True, dim=-1)[1].detach().clone().cpu()
                        # loss_hamming = hamming_distance(st_sort, tc_sort)
                        
                        # hamming_sum += loss_hamming
                        # Ranking Loss
                        
                        # Loss_ranking = 0
                        # for h in range(seq_length):
                        #     for idx in range(0, seq_length-1):
                        #         for jdx in range(1, seq_length):
                        #             p = (student_prob_plt[h][idx] - student_prob_plt[h][jdx])*(torch.sgn(teacher_prob_plt[h][idx] - teacher_prob_plt[h][jdx]))
                        #             Loss_ranking += max(0,  - p.item())
                        
                        # Loss_ranking_sum += Loss_ranking

                        # Hamming Distance

                        # st_sort = student_prob.sort(descending=True, dim=3)[1].detach().clone().cpu()
                        # tc_sort = teacher_prob.sort(descending=True, dim=3)[1].detach().clone().cpu()
                        # loss_hamming = hamming_distance(st_sort, tc_sort)


                if task_name=='cola':
                    #summaryWriter.add_scalar('mcc',result['mcc'],global_step)
                    if run is not None:
                        run["metrics/mcc"].log(value=result['mcc'], step=global_step)
                    logger.info(f"Eval Result is {result['mcc']}")
                elif task_name in ['sst-2','mnli','mnli-mm','qnli','rte','wnli']:
                    #summaryWriter.add_scalar('acc',result['acc'],global_step)
                    if run is not None:
                        run["metrics/acc"].log(value=result['acc'],step=global_step)
                    logger.info(f"Eval Result is {result['acc']}")
                elif task_name in ['mrpc','qqp']:
                    # summaryWriter.add_scalars('performance',{'acc':result['acc'],
                    #                             'f1':result['f1'],
                    #                             'acc_and_f1':result['acc_and_f1']},global_step)
                    if run is not None:
                        run["metrics/acc_and_f1"].log(value=result['acc_and_f1'],step=global_step)
                    logger.info(f"Eval Result is {result['acc']}, {result['f1']}")
                else:
                    #summaryWriter.add_scalar('corr',result['corr'],global_step)
                    if run is not None:
                        run["metrics/corr"].log(value=result['corr'],step=global_step)
                    logger.info(f"Eval Result is {result['corr']}")

                # Save Model
                save_model = False

                if task_name in acc_tasks and result['acc'] > best_dev_acc:
                    if task_name in ['sst-2','mnli','qnli','rte']:
                        previous_best = f"acc:{result['acc']}"
                    elif task_name in ['mrpc','qqp']:
                        previous_best = f"f1/acc:{result['f1']}/{result['acc']}"
                    best_dev_acc = result['acc']
                    save_model = True

                if task_name in corr_tasks and result['corr'] > best_dev_acc:
                    previous_best = f"pearson/spearmanr:{result['pearson']}/{result['spearmanr']}"
                    best_dev_acc = result['corr']
                    save_model = True

                if task_name in mcc_tasks and result['mcc'] > best_dev_acc:
                    previous_best = f"mcc:{result['mcc']}"
                    best_dev_acc = result['mcc']
                    save_model = True

                if save_model:
                    # Test mnli-mm
                    if task_name == "mnli":
                        result = do_eval(student_model, 'mnli-mm', mm_eval_dataloader,
                                            device, output_mode, mm_eval_labels, num_labels)
                        previous_best+= f"mm-acc:{result['acc']}"
                    logger.info(fp32_performance)
                    logger.info(previous_best)
                    if args.save_fp_model:
                        logger.info("***** Save full precision model *****")
                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                        output_config_file = os.path.join(output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(output_dir)
                    if args.save_quantized_model:
                        logger.info("====> Save quantized model *****")

                        output_quant_dir = os.path.join(output_dir, 'quant')
                        if not os.path.exists(output_quant_dir):
                            os.makedirs(output_quant_dir)
                        
                        output_quant_dir = os.path.join(output_quant_dir, args.exp_name)
                        if not os.path.exists(output_quant_dir):
                            os.makedirs(output_quant_dir)
                        
                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                        quant_model = copy.deepcopy(model_to_save)
                        # for name, module in quant_model.named_modules():
                        #     if hasattr(module,'weight_quantizer'):
                        #         module.qweight = module.weight_quantizer(module.weight, True)
                                
                        output_model_file = os.path.join(output_quant_dir, WEIGHTS_NAME)
                        output_config_file = os.path.join(output_quant_dir, CONFIG_NAME)

                        torch.save(quant_model.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(output_quant_dir)


if __name__ == "__main__":
    main()
