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
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss, MSELoss

from transformer import BertForSequenceClassification,WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling_quant import BertForSequenceClassification as QuantBertForSequenceClassification
from transformer import BertTokenizer
from transformer import BertAdam
from transformer import BertConfig
from utils_glue import *

import matplotlib.pyplot as plt

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

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

def do_eval(model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels, teacher=False):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for _,batch_ in enumerate(eval_dataloader):
        batch_ = tuple(t.to(device) for t in batch_)
        
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_
            if not teacher:
            logits, _, _ = model(input_ids, segment_ids, input_mask)

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
                        action='store_true',
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
                        action='store_true',
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

    logger.info("==================================================>Setup...")
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
    
    if args.student_model is None:
        #args.student_model = os.path.join("output", "BERT_base",task_name.upper())
        args.student_model = os.path.join(args.model_dir, "FFN")
    if args.teacher_model is None:
        args.teacher_model = os.path.join("output", "BERT_base",task_name.upper())
        #args.teacher_model = os.path.join(args.model_dir, "FFN")

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
        "cola": {"max_seq_length": 64,"batch_size":16,"eval_step":50},
        "mnli": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "mrpc": {"max_seq_length": 128,"batch_size":32,"eval_step":100},
        "sst-2": {"max_seq_length": 64,"batch_size":32,"eval_step":200},
        "sts-b": {"max_seq_length": 128,"batch_size":32,"eval_step":50},
        "qqp": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "qnli": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "rte": {"max_seq_length": 128,"batch_size":32,"eval_step":100}
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
    
    # ================================================================================  #
    # Dataset Setup
    # ================================================================================ #
    
    if args.aug_train:
        try:
            train_file = os.path.join(processed_data_dir,'aug_data.pkl')
            train_features = pickle.load(open(train_file,'rb'))
        except:
            train_examples = processor.get_aug_examples(data_dir)
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

    num_train_epochs = 3 if not args.aug_train else 1
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
    logger.info("==================================================>Build Teacher Model..")
    teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_model)

    teacher_model.to(device)
    teacher_model.eval()
    if n_gpu > 1:
        teacher_model = torch.nn.DataParallel(teacher_model, device_ids=range(n_gpu))
    logger.info("==================================================>Teacher Model Evaluation..")
    result = do_eval(teacher_model, task_name, eval_dataloader,
                    device, output_mode, eval_labels, num_labels, teacher=True)
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
    logger.info("==================================================>Build Student Model...")
    student_config = BertConfig.from_pretrained(args.teacher_model, 
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
                                                gradient_scaling = args.gradient_scaling)
    
    student_model = QuantBertForSequenceClassification.from_pretrained(args.student_model, config = student_config, num_labels=num_labels)
    student_model.to(device)
    logger.info("==================================================>Student Model Evaluation...")
    result = do_eval(student_model, task_name, eval_dataloader,
                    device, output_mode, eval_labels, num_labels, teacher=False)
    logger.info(result)


if __name__ == "__main__":
    main()
