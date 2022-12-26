from __future__ import absolute_import, division, print_function

import pprint
import argparse
import logging
import os
import random
import sys
import pickle
import copy
import collections
import math

import numpy as np
import numpy
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset

from torch.nn import CrossEntropyLoss, MSELoss

from transformer import BertForSequenceClassification,WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling_quant import BertForSequenceClassification as QuantBertForSequenceClassification
from transformer import BertTokenizer
from transformer import BertAdam
from transformer import BertConfig
from transformer import QuantizeLinear, BertSelfAttention, FP_BertSelfAttention
from utils_glue import *
from bertviz import model_view

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn.functional as F
import time

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

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[index] = token
            index += 1
    return vocab

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
            device, output_mode, eval_labels, num_labels, teacher_model=None):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for batch_ in eval_dataloader:
        batch_ = tuple(t.to(device) for t in batch_)
        
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

            # teacher attnmap test
            if teacher_model is not None:
                teacher_logits, teacher_atts, teacher_reps, teacher_probs, teacher_values = teacher_model(input_ids, segment_ids, input_mask)
                logits, student_atts, student_reps, student_probs, student_values  = model(input_ids, segment_ids, input_mask)
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

    # Options for training setting
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
                        help="Student model directory.")
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
    parser.add_argument('--save_fp_model',
                        action='store_true',
                        help="Whether to save fp32 model")
    parser.add_argument('--save_quantized_model',
                        default=False, type=str2bool,
                        help="Whether to save quantized model")
    parser.add_argument("--weight_bits",
                        default=2,
                        type=int,
                        choices=[2,4,8],
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
    parser.add_argument('--weight_quant',
                        default=True, type=str2bool,
                        help="Whether to quantize activation")
    parser.add_argument('--neptune',
                        default=True, type=str2bool,
                        help="neptune logging option")
    parser.add_argument('--aug_train',
                        default =False, type=str2bool,
                        help="Whether to use augmented data or not")
    parser.add_argument("--aug_N",
                        default=30,
                        type=int,
                        help="Data Augmentation N Number")
    parser.add_argument("--exp_name",
                        default="",
                        type=str,
                        help="Output Directory Name")

    # Options for Distillation
    parser.add_argument('--pred_distill',
                        default =False, type=str2bool,
                        help="prediction distill option")

    parser.add_argument('--attn_distill',
                        default =True, type=str2bool,
                        help="attention Score Distill Option")

    parser.add_argument('--rep_distill',
                        default =True, type=str2bool,
                        help="Transformer Layer output Distill Option")

    parser.add_argument('--attnmap_distill',
                        default =True, type=str2bool,
                        help="attention Map Distill Option")

    parser.add_argument('--context_distill',
                        default =True, type=str2bool,
                        help="Context Value Distill Option")

    parser.add_argument('--output_distill',
                        default =False, type=str2bool,
                        help="Context Value Distill Option")
    
    parser.add_argument('--sa_output_distill',
                        default =False, type=str2bool,
                        help="MSA output Distill Option")
    
    parser.add_argument('--gt_loss',
                        default =True, type=str2bool,
                        help="Ground Truth Option")
    
    parser.add_argument('--bert',
                        default ="base", type=str,
                        help="which bert model to be use (base, large)"
    )
    parser.add_argument("--map_coeff",
                        default=1,
                        type=float,
                        help="Attention Map Loss Coefficient")
    parser.add_argument("--output_coeff",
                        default=1,
                        type=float,
                        help="Attention Output Loss Coefficient")

    args = parser.parse_args() 
    
    # ================================================================================  #
    # Logging setup
    # ================================================================================ #
    # Use Neptune for logging
    if args.neptune:
        import neptune.new as neptune        
        # Neptune Init
        run = neptune.init(project='' + args.task_name.upper(),
                    api_token='')
    else:
        run = None

    # ================================================================================  #
    # Load Directory
    # ================================================================================ #
    
    # Exp Name
    exp_name = args.exp_name 

    exp_name += f"_{args.bert}"

    if args.attn_distill:
        exp_name += "_S"
    if args.attnmap_distill:
        exp_name += "_M"
    if args.context_distill:
        exp_name += "_C"
    if args.output_distill:
        exp_name += "_O"
    if args.sa_output_distill:
        exp_name += "_SA"
    exp_name += f"_{args.seed}"            
    
    args.exp_name = exp_name
    
    if args.aug_train:
        logger.info(f'DA QAT')        
        
    logger.info(f'EXP SET: {exp_name}')
    logger.info(f'TASK: {args.task_name}')
    logger.info(f'MAP-COEFF: {args.map_coeff}')
    logger.info(f'OUTPUT-COEFF: {args.output_coeff}')
    logger.info(f"SIZE: {args.bert}")
    logger.info(f"SEED: {args.seed}")
    logger.info(f'EPOCH: {args.num_train_epochs}')
    
    # logger.info('The args: {}'.format(args))
    
    # GLUE Dataset Setting
    task_name = args.task_name.lower()
    data_dir = os.path.join(args.data_dir,task_name)
    processed_data_dir = os.path.join(data_dir,'preprocessed')
    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)
    
    # BERT Large Option
    if args.bert == "large":
        args.model_dir = os.path.join(args.model_dir, "BERT_large")
        args.output_dir = os.path.join(args.output_dir, "BERT_large")
    
    # Model Save Directory
    output_dir = os.path.join(args.output_dir,task_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    if args.save_quantized_model:
        output_quant_dir = os.path.join(output_dir, 'exploration')
        if not os.path.exists(output_quant_dir):
            os.mkdir(output_quant_dir)

        if not os.path.exists(output_quant_dir):
            os.makedirs(output_quant_dir)
        
        output_quant_dir = os.path.join(output_quant_dir, args.exp_name)
        if not os.path.exists(output_quant_dir):
            os.makedirs(output_quant_dir)

    # ================================================================================  #
    # Load Pths
    # ================================================================================ #
    # Load task-specific fine-tuned model file
    args.student_model = os.path.join(args.model_dir,task_name) 
    args.teacher_model = os.path.join(args.model_dir,task_name)
    
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
        "cola": {"max_seq_length": 64,"batch_size":16,"eval_step": 400 if args.aug_train else 50}, # No Aug : 50 Aug : 400
        "mnli": {"max_seq_length": 128,"batch_size":32,"eval_step":8000},
        "mrpc": {"max_seq_length": 128,"batch_size":32,"eval_step":50},
        "sst-2": {"max_seq_length": 64,"batch_size":32,"eval_step":100},
        "sts-b": {"max_seq_length": 128,"batch_size":32,"eval_step":100},
        "qqp": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "qnli": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "rte": {"max_seq_length": 128,"batch_size":32,"eval_step":100 if args.aug_train else 20}
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
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=True)
    # save vocab file for logging
    tokenizer.save_vocabulary("./")

    # ================================================================================  #
    # Dataset Setup (with DA)
    # ================================================================================ #
    if args.aug_train: # Data Augmentation
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

    num_train_epochs = args.num_train_epochs 
    num_train_optimization_steps = math.ceil(len(train_features) / args.batch_size) * num_train_epochs
        
    train_data, _ = get_tensor_data(output_mode, train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    
    # Test Data load
    try:
        test_file = train_file = os.path.join(processed_data_dir,'test.pkl')
        test_features = pickle.load(open(test_file,'rb'))
    except:
        test_examples = processor.get_test_examples(data_dir)
        test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        with open(test_file, 'wb') as f:
                pickle.dump(test_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Dev Data load
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
        # logger.info("  Num examples = %d", len(mm_eval_features))

        mm_eval_sampler = SequentialSampler(mm_eval_data)
        mm_eval_dataloader = DataLoader(mm_eval_data, sampler=mm_eval_sampler,
                                        batch_size=args.batch_size)


    # ================================================================================ #
    # Build Teacher Model
    # ================================================================================ # 
    teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_model, num_labels=num_labels)
    
    teacher_model.to(device)
    teacher_model.eval()
    
    if n_gpu > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)
    
    result = do_eval(teacher_model, task_name, eval_dataloader,
                    device, output_mode, eval_labels, num_labels)
    # logger.info(f"Teacher Model Accuracy : {result}")
    
    # ================================================================================  #
    # Save Teacher Model Peroformance for KD Training
    # ================================================================================ #
    if task_name in acc_tasks:
        if task_name in ['sst-2','mnli','qnli','rte']:
            fp32_performance = f"acc:{result['acc']}"
            fp32_score = result['acc']
        elif task_name in ['mrpc','qqp']:
            fp32_performance = f"f1/acc:{result['f1']}/{result['acc']} avg : {(result['f1'] + result['acc'])*50}"
            fp32_score = (result['f1'] + result['acc'])*50
    if task_name in corr_tasks:
        fp32_performance = f"pearson/spearmanr:{result['pearson']}/{result['spearmanr']} corr:{result['corr']}"
        fp32_score = result['corr']*100

    if task_name in mcc_tasks:
        fp32_performance = f"mcc:{result['mcc']}"
        fp32_score = result['mcc']

    if task_name == "mnli":
        result = do_eval(teacher_model, 'mnli-mm', mm_eval_dataloader,
                            device, output_mode, mm_eval_labels, num_labels)
        # logger.info(result)
        fp32_performance += f"  mm-acc:{result['acc']}"
        fp32_score = result['acc']
    fp32_performance = task_name +' fp32   ' + fp32_performance
    
    # ================================================================================  #
    # Build Student Model
    # ================================================================================ #
    student_config = BertConfig.from_pretrained(args.student_model, 
                                                quantize_act=args.act_quant,
                                                quantize_weight=args.weight_quant,
                                                weight_bits = args.weight_bits,
                                                input_bits = args.input_bits,
                                                clip_val = args.clip_val,
                                                )
    
    student_model = QuantBertForSequenceClassification.from_pretrained(args.student_model, config = student_config, num_labels=num_labels)
    
    student_model.to(device)
    
    for name, module in student_model.named_modules():
        if isinstance(module, QuantizeLinear):
            module.act_flag = args.act_quant
            module.weight_flag = args.weight_quant      
    
    # ================================================================================  #
    # Training Setting
    # ================================================================================ #
    if n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)
    param_optimizer = list(student_model.named_parameters())
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    schedule = 'warmup_linear'
    optimizer = BertAdam(optimizer_grouped_parameters,
                            schedule=schedule,
                            lr=args.learning_rate,
                            warmup=0.1,
                            t_total=num_train_optimization_steps)
    
    loss_mse = MSELoss()
    norm_func = torch.linalg.norm
    loss_cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    
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

    # logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_features))
    # logger.info("  Batch size = %d", args.batch_size)
    # logger.info("  Num steps = %d", num_train_optimization_steps)
    
    # Loss Init AverageMeter
    l_gt_loss = AverageMeter()
    l_attmap_loss = AverageMeter()
    l_att_loss = AverageMeter()
    l_rep_loss = AverageMeter()
    l_cls_loss = AverageMeter()
    l_output_loss = AverageMeter()
    l_sa_output_loss = AverageMeter()
    l_context_loss = AverageMeter()
    l_loss = AverageMeter()
    
    for epoch_ in range(int(num_train_epochs)):
    
        # for batch in tqdm(train_dataloader,desc=f"Epoch_{epoch_}", mininterval=0.01, ascii=True, leave=False):
        for batch in train_dataloader:

            student_model.train()
            
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
            
            # tmp loss init
            att_loss = 0.
            attmap_loss = 0.
            rep_loss = 0.
            cls_loss = 0.
            attscore_loss = 0.
            output_loss = 0.
            sa_output_loss = 0.
            context_loss = 0.
            loss = 0.
            
            with torch.no_grad():
                teacher_logits, teacher_atts, teacher_reps, teacher_probs, teacher_attn_blocks = teacher_model(input_ids, segment_ids, input_mask)
            
            # attn_blocks -> 0: Layer Context, 1: Attention Sub-layer Output, 2: Self-Attention Output 
            student_logits, student_atts, student_reps, student_probs, student_attn_blocks = student_model(input_ids, segment_ids, input_mask)

            if args.gt_loss:
                if output_mode == "classification":
                    lprobs = torch.nn.functional.log_softmax(student_logits, dim=-1)
                    loss = torch.nn.functional.nll_loss(lprobs, label_ids, reduction='sum')
                elif output_mode == "regression":
                    loss = loss_mse(student_logits, teacher_logits)
                l_gt_loss.update(loss.item())
                
            # Prediction Loss
            if args.pred_distill:
                if output_mode == "classification":
                    cls_loss = soft_cross_entropy(student_logits,teacher_logits)
                elif output_mode == "regression":
                    cls_loss = MSELoss()(student_logits, teacher_logits)
                else:
                    cls_loss = soft_cross_entropy(student_logits,teacher_logits)
                l_cls_loss.update(cls_loss.item())

            # Attention Context Loss
            if args.context_distill:
                for i, (student_attn_block, teacher_attn_block) in enumerate(zip(student_attn_blocks, teacher_attn_blocks)):    
                    tmp_loss = MSELoss()(student_attn_block[0], teacher_attn_block[0]) 
                    context_loss += tmp_loss
                l_context_loss.update(context_loss.item())
            
            # Attention Output Loss (Proposed)
            if args.output_distill:
                for i, (student_attn_block, teacher_attn_block) in enumerate(zip(student_attn_blocks, teacher_attn_blocks)):    
                    tmp_loss = MSELoss()(student_attn_block[1], teacher_attn_block[1]) 
                    output_loss += tmp_loss
                l_output_loss.update(output_loss.item())
            
            # SA Output Loss
            if args.sa_output_distill:
                for i, (student_attn_block, teacher_attn_block) in enumerate(zip(student_attn_blocks, teacher_attn_blocks)):    
                    tmp_loss = MSELoss()(student_attn_block[2], teacher_attn_block[2]) 
                    sa_output_loss += tmp_loss
                l_sa_output_loss.update(sa_output_loss.item())
            
            # Attention Score Loss (TernaryBERT)
            if args.attn_distill:
                for i, (student_att, teacher_att) in enumerate(zip(student_atts, teacher_atts)):    
                            
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to("cuda"),
                                                student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to("cuda"),
                                                teacher_att)

                    tmp_loss = MSELoss()(student_att, teacher_att)
                    attscore_loss += tmp_loss
                l_att_loss.update(attscore_loss.item())

            # Attention Map Loss (Proposed)
            if args.attnmap_distill:
            
                BATCH_SIZE = student_probs[0].shape[0]
                NUM_HEADS = student_probs[0].shape[1]
                MAX_SEQ = student_probs[0].shape[2]
                
                mask = torch.zeros(BATCH_SIZE, NUM_HEADS, MAX_SEQ, MAX_SEQ, dtype=torch.float32)
                mask_seq = []
                
                for sent in range(BATCH_SIZE):
                    s = seq_lengths[sent]
                    mask[sent, :, :s, :s] = 1.0
                
                mask = mask.to("cuda")

                for i, (student_prob, teacher_prob) in enumerate(zip(student_probs, teacher_probs)):            
                        
                    # KLD(teacher || student)
                    # = sum (p(t) log p(t)) - sum(p(t) log p(s))
                    # = (-entropy) - (-cross_entropy)
                    
                    student = torch.clamp_min(student_prob, 1e-8)
                    teacher = torch.clamp_min(teacher_prob, 1e-8)

                    # p(t) log p(s) = negative cross entropy
                    neg_cross_entropy = teacher * torch.log(student) * mask
                    neg_cross_entropy = torch.sum(neg_cross_entropy, dim=-1)  # (b, h, s, s) -> (b, h, s)
                    neg_cross_entropy = torch.sum(neg_cross_entropy, dim=-1) / seq_lengths.view(-1, 1)  # (b, h, s) -> (b, h)

                    # p(t) log p(t) = negative entropy
                    neg_entropy = teacher * torch.log(teacher) * mask
                    neg_entropy = torch.sum(neg_entropy, dim=-1)  # (b, h, s, s) -> (b, h, s)
                    neg_entropy = torch.sum(neg_entropy, dim=-1) / seq_lengths.view(-1, 1)  # (b, h, s) -> (b, h)

                    kld_loss = neg_entropy - neg_cross_entropy
                    
                    kld_loss_mean = torch.mean(kld_loss)

                    # Other Option (Cosine Similarity, MSE Loss)
                    # attnmap_mse_loss = loss_mse(student, teacher)
                    # kld_loss_sum = torch.nn.functional.cosine_similarity(student, teacher, -1).mean()
                            
                    attmap_loss += kld_loss_mean
                
                l_attmap_loss.update(attmap_loss.item())
            
            # Rep Distill
            if args.rep_distill:
                for i, (student_rep, teacher_rep) in enumerate(zip(student_reps, teacher_reps)):
                    tmp_loss = MSELoss()(student_rep, teacher_rep)
                    rep_loss += tmp_loss
                l_rep_loss.update(rep_loss.item())

            loss += cls_loss + rep_loss + (attmap_loss * args.map_coeff) + (output_loss * args.output_coeff) + sa_output_loss + attscore_loss + context_loss
            l_loss.update(loss.item())
            

            if n_gpu > 1:
                loss = loss.mean()           
                
            # Zero Step Loss Update
            if global_step == 0: 
                if run is not None:           
                    run["loss/total_loss"].log(value=l_loss.avg, step=global_step)
                    run["loss/gt_loss_loss"].log(value=l_gt_loss.avg, step=global_step)
                    run["loss/att_loss_loss"].log(value=l_att_loss.avg, step=global_step)
                    run["loss/rep_loss_loss"].log(value=l_rep_loss.avg, step=global_step)
                    run["loss/cls_loss_loss"].log(value=l_cls_loss.avg, step=global_step)
                    run["loss/output_loss_loss"].log(value=l_output_loss.avg, step=global_step)
                    run["loss/attmap_loss_loss"].log(value=l_attmap_loss.avg, step=global_step)
                    
                    run["metrics/lr"].log(value=optimizer.get_lr()[0], step=global_step)
 
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()
            
            global_step += 1

            # ================================================================================  #
            #  Evaluation
            # ================================================================================ #

            if global_step % args.eval_step == 0 or global_step == num_train_optimization_steps-1: # period or last step
                # logger.info("***** Running evaluation *****")
                # logger.info("{} step of {} steps".format(global_step, num_train_optimization_steps))
                
                # if previous_best is not None:
                    # logger.info(f"{fp32_performance}")
                    # logger.info(f"==> Previous Best = {previous_best}")

                student_model.eval()
                result = do_eval(student_model, task_name, eval_dataloader,
                                    device, output_mode, eval_labels, num_labels, teacher_model=teacher_model)
            
                result['global_step'] = global_step
                result['cls_loss'] = l_cls_loss.avg
                result['att_loss'] = l_att_loss.avg
                result['rep_loss'] = l_rep_loss.avg
                result['loss'] = l_loss.avg
                
                if run is not None:
                    
                    run["loss/total_loss"].log(value=l_loss.avg, step=global_step)
                    run["loss/gt_loss_loss"].log(value=l_gt_loss.avg, step=global_step)
                    run["loss/att_loss_loss"].log(value=l_att_loss.avg, step=global_step)
                    run["loss/rep_loss_loss"].log(value=l_rep_loss.avg, step=global_step)
                    run["loss/cls_loss_loss"].log(value=l_cls_loss.avg, step=global_step)
                    run["loss/output_loss_loss"].log(value=l_output_loss.avg, step=global_step)
                    run["loss/sa_output_loss_loss"].log(value=l_sa_output_loss.avg, step=global_step)
                    run["loss/attmap_loss_loss"].log(value=l_attmap_loss.avg, step=global_step)
                    run["metrics/lr"].log(value=optimizer.get_lr()[0], step=global_step)

                if task_name=='cola':
                    eval_score = result["mcc"]
                    if run is not None:
                        run["metrics/mcc"].log(value=result['mcc'], step=global_step)

                    eval_result = result["mcc"]  
                    # logger.info(f"Eval Result is {result['mcc']}")
                elif task_name in ['sst-2','mnli','mnli-mm','qnli','rte','wnli']:
                    eval_score = result["acc"]
                    if run is not None:
                        run["metrics/acc"].log(value=result['acc'],step=global_step)
                        
                    # logger.info(f"Eval Result is {result['acc']}")
                    eval_result = result["acc"]
                elif task_name in ['mrpc','qqp']:
                    eval_score = result["acc_and_f1"]
                    if run is not None:
                        run["metrics/acc_and_f1"].log(value=result['acc_and_f1'],step=global_step)
                        
                    # logger.info(f"Eval Result is {result['acc']}, {result['f1']}")
                    eval_result = result["acc_and_f1"]
                else:
                    eval_score = result["corr"]
                    if run is not None:
                        run["metrics/corr"].log(value=result['corr'],step=global_step)
                        
                    # logger.info(f"Eval Result is {result['corr']}")
                    eval_result = result["corr"]                
                
                # Save Model
                save_model = False

                if task_name in acc_tasks and result['acc'] > best_dev_acc:
                    if task_name in ['sst-2','mnli','qnli','rte']:
                        previous_best = f"{result['acc']*100}"
                    elif task_name in ['mrpc','qqp']:
                        previous_best = f"{(result['f1'] + result['acc'])*50}"
                    best_dev_acc = result['acc']
                    save_model = True

                if task_name in corr_tasks and result['corr'] > best_dev_acc:
                    previous_best = f"{result['corr']*100}"
                    best_dev_acc = result['corr']
                    save_model = True

                if task_name in mcc_tasks and result['mcc'] > best_dev_acc:
                    previous_best = f"{result['mcc']*100}"
                    best_dev_acc = result['mcc']
                    save_model = True

                if save_model:
                    # logger.info("====> Best Score ")
                    # Test mnli-mm
                    if task_name == "mnli":
                        result = do_eval(student_model, 'mnli-mm', mm_eval_dataloader,
                                            device, output_mode, mm_eval_labels, num_labels, teacher_model=teacher_model)
                        previous_best+= f"mm-acc:{result['acc']}"

                    if args.save_fp_model:
                        # logger.info("***** Save full precision model *****")
                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                        output_config_file = os.path.join(output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(output_dir)
                    
                    if args.save_quantized_model:
                        # logger.info("====> Save quantized model *****")

                        output_quant_dir = os.path.join(output_dir, 'exploration')
                        if not os.path.exists(output_quant_dir):
                            os.mkdir(output_quant_dir)

                        if not os.path.exists(output_quant_dir):
                            os.makedirs(output_quant_dir)
                        
                        output_quant_dir = os.path.join(output_quant_dir, args.exp_name)
                        if not os.path.exists(output_quant_dir):
                            os.makedirs(output_quant_dir)
                        
                        # uncomment if you want to save your quantized model
                        
                        # model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                        # quant_model = copy.deepcopy(model_to_save)
                            
                        # output_model_file = os.path.join(output_quant_dir, WEIGHTS_NAME)
                        # output_config_file = os.path.join(output_quant_dir, CONFIG_NAME)

                        # torch.save(quant_model.state_dict(), output_model_file)
                        # model_to_save.config.to_json_file(output_config_file)
                        # tokenizer.save_vocabulary(output_quant_dir)
                        
    logger.info(f"==> Previous Best = {previous_best}")
    logger.info(f"==> Last Result = {result}")
    
    # Save Best Score
    if args.save_quantized_model:
        best_txt = os.path.join(output_quant_dir, "best_info.txt")
        last_txt = os.path.join(output_quant_dir, "last_info.txt")
        with open(best_txt, "w") as f_w:
            f_w.write(previous_best)
        with open(last_txt, "w") as f_w:
            f_w.write(f"{eval_result*100}")

if __name__ == "__main__":
    main()