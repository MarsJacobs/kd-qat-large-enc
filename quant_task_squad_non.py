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

from transformer import BertForQuestionAnswering,WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling_quant import BertForQuestionAnswering as QuantBertForQuestionAnswering
from transformer import BertTokenizer
from transformer import BertAdam
from transformer import BertConfig
from transformer import QuantizeLinear, QuantizeAct, BertSelfAttention, FP_BertSelfAttention, ClipLinear
from utils_glue import *
from bertviz import model_view

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn.functional as F

from utils_squad import *

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

def cv_initialize(model, loader, ratio, device):
    
    def initialize_hook(module, input, output):
        if isinstance(module, (QuantizeLinear, QuantizeAct, ClipLinear)):
            """KDLSQ-BERT ACT Quant init Method
            Ref: https://arxiv.org/abs/2101.05938
            """
            if not isinstance(input, torch.Tensor):
                input = input[0]
        
            n = torch.numel(input)
            input_sorted, index = torch.sort(input.reshape(-1), descending=False)
            
            index_min = torch.round(ratio * n / 2)
            index_max = n - index_min
            
            s_init = (input_sorted[int(index_min)].to(device), input_sorted[int(index_max)].to(device))
            
            # fig, [ax1, ax2] = plt.subplots(1,2, figsize=(12, 4))            
            
            # sns.histplot(data=input.reshape(-1).detach().cpu().numpy(), kde = True, bins=100, ax=ax1)
            # sns.rugplot(data=input.reshape(-1).detach().cpu().numpy(), ax=ax1)
            # sns.histplot(data=output.reshape(-1).detach().cpu().numpy(), kde = True, bins=100, ax=ax2)
            # sns.rugplot(data=output.reshape(-1).detach().cpu().numpy(), ax=ax2)

            
            # fig, [ax1, ax2] = plt.subplots(1,2, figsize=(12, 4))            
            
            # sns.distplot(input.reshape(-1).detach().cpu().numpy() , hist = True, rug = True, kde = True, bins=100, norm_hist=False, kde_kws=dict(linewidth=0.5), rug_kws=dict(linewidth=0.5), ax=ax1)
            # sns.distplot(output.reshape(-1).detach().cpu().numpy() , hist = True, rug = True, kde = True, bins=100, norm_hist=False, kde_kws=dict(linewidth=0.5), rug_kws=dict(linewidth=0.5), ax=ax2)
            # # plt.axvline(x=s_init[0].detach().cpu().numpy(), color='r', linestyle='--')
            # # plt.axvline(x=s_init[1].detach().cpu().numpy(), color='r', linestyle='--')

            # ax1.set_xlabel("Input Activation")
            # ax2.set_xlabel("Output Activation")
            
            # ax1.set_ylabel("Density")
            # ax2.set_ylabel("Density")

            # ax1.set_title(f"{module.name} Input ACT histogram")
            # ax2.set_title(f"{module.name} Output ACT histogram")
            # plt.savefig(f"plt_storage/hook_inputs/sst-2-fp/{module.name}.png")
            
            # plt.close(fig)

            # logger.info(f"{module.name} : min {s_init[0].item()} max {s_init[1].item()}") 
            module.clip_initialize(s_init)
    
    hooks = []

    for name, module in model.named_modules():
        hook = module.register_forward_hook(initialize_hook)
        hooks.append(hook)
    
    model.train()
    model.to(device)
    
    for step, batch in enumerate(loader):
        batch = tuple(t.to("cuda") for t in batch)
        input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch        
        with torch.no_grad():
            student_logits, student_atts, student_reps, student_probs, student_values = model(input_ids, segment_ids, input_mask, teacher_probs=None)
        break
    
    for hook in hooks:
        hook.remove()

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
            #vocab[token] = index
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

    parser.add_argument('--save_fp_model',
                        action='store_true',
                        help="Whether to save fp32 model")
                        
    parser.add_argument('--save_quantized_model',
                        default=False, type=str2bool,
                        help="Whether to save quantized model")
    
    parser.add_argument('--log_map',
                        default=False, type=str2bool,
                        )
    
    parser.add_argument('--log_metric',
                        default=False, type=str2bool,
                        )

    parser.add_argument("--weight_bits",
                        default=2,
                        type=int,
                        choices=[2,4,8],
                        help="Quantization bits for weight.")
    parser.add_argument("--input_bits",
                        default=8,
                        type=int,
                        help="Quantization bits for activation.")
    
    parser.add_argument("--tc_top_k",
                        default=3,
                        type=int,
                        help="Top-K Coverage")

    parser.add_argument("--gpus",
                        default=1,
                        type=int,
                        help="Number of GPUs to use")
    parser.add_argument("--clip_val",
                        default=2.5,
                        type=float,
                        help="Initial clip value.")

    parser.add_argument('--version_2_with_negative',
                        default=False, type=str2bool,
                        help="Squadv2.0 if true else Squadv1.1")

    parser.add_argument('--act_quant',
                        default=True, type=str2bool,
                        help="Whether to quantize activation")

    parser.add_argument('--weight_quant',
                        default=True, type=str2bool,
                        help="Whether to quantize activation")
    
    parser.add_argument('--parks',
                        default=False, type=str2bool,
                        help="Whether to quantize activation")

    parser.add_argument('--stop_grad',
                        default=False, type=str2bool,
                        help="Whether to quantize activation")
    
    parser.add_argument('--qk_FP',
                        default=False, type=str2bool,
                        )
    
    parser.add_argument('--qkv_FP',
                        default=False, type=str2bool,
                        )
    
    parser.add_argument('--neptune',
                        default=True, type=str2bool,
                        help="neptune logging option")
    
    #MSKIM Quantization Option
    parser.add_argument("--quantizer",
                        default="ternary",
                        type=str,
                        help="Quantization Method")

    parser.add_argument("--act_quantizer",
                        default="pact",
                        type=str,
                        help="ACT Quantization Method")

    #MSKIM Quantization Range Option
    parser.add_argument('--quantize',
                        default =True, type=str2bool,
                        help="Whether to quantize student model")

    parser.add_argument('--ffn_1',
                        default =True, type=str2bool,
                        help="Whether to quantize Feed Forward Network")
    
    parser.add_argument('--ffn_2',
                        default =True, type=str2bool,
                        help="Whether to quantize Feed Forward Network")
    
    parser.add_argument('--qkv',
                        default =True, type=str2bool,
                        help="Whether to quantize Query, Key, Value Mapping Weight Matrix")
    
    parser.add_argument('--emb',
                        default =True, type=str2bool,
                        help="Whether to quantize Embedding Layer")

    parser.add_argument('--cls',
                        default =True, type=str2bool,
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

    parser.add_argument('--gradient_scaling',
                        default =1, type=float,
                        help="LSQ gradient scaling")

    parser.add_argument("--layer_num",
                        default=-1,
                        type=int,
                        help="Number of layer to quantize (-1 : Quantize every layer")
    
    parser.add_argument("--layer_thres_num",
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
    
    parser.add_argument("--training_type",
                        default="qat_normal",
                        type=str,
                        help="QAT Method")
    
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

    parser.add_argument("--sm_temp",
                        default=0.4,
                        type=float,
                        help="LSM Temperature Value")

    parser.add_argument("--index_ratio",
                        default=0.05,
                        type=float,
                        help="KDLSQ index ratio")
    
    parser.add_argument("--other_lr",
                        default=0.0001,
                        type=float,
                        help="other param lr")

    parser.add_argument("--attnmap_coeff",
                        default=1,
                        type=float,
                        help="attnmap loss coeff")
    
    parser.add_argument("--word_coeff",
                        default=1,
                        type=float,
                        help="attn wrod loss coeff")
    
    parser.add_argument("--cls_coeff",
                        default=1,
                        type=float,
                        help="cls loss coeff")
    
    parser.add_argument("--att_coeff",
                        default=1,
                        type=float,
                        help="att loss coeff")
    
    parser.add_argument("--val_coeff",
                        default=1,
                        type=float,
                        help="val loss coeff")
    
    parser.add_argument("--context_coeff",
                        default=1,
                        type=float,
                        help="context loss coeff")
    
    parser.add_argument("--output_coeff",
                        default=1,
                        type=float,
                        help="output loss coeff")

    parser.add_argument("--rep_coeff",
                        default=1,
                        type=float,
                        help="rep loss coeff")

    parser.add_argument("--aug_N",
                        default=30,
                        type=int,
                        help="Data Augmentation N Number")

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
    
    parser.add_argument('--gt_loss',
                        default =True, type=str2bool,
                        help="Ground Truth Option")
    
    parser.add_argument('--word_distill',
                        default =True, type=str2bool,
                        help="Ground Truth Option")
    
    parser.add_argument('--val_distill',
                        default =False, type=str2bool,
                        help="attention Map Distill Option")

    parser.add_argument('--teacher_attnmap',
                        default =False, type=str2bool,
                        help="Teacher Intervention Option (Attention Map)")

    parser.add_argument('--teacher_context',
                        default =False, type=str2bool,
                        help="Teacher Intervention Option (Context)")
    
    parser.add_argument('--teacher_input',
                        default =False, type=str2bool,
                        help="Teacher Intervention Option (Input)")
    
    parser.add_argument('--map',
                        default =False, type=str2bool,
                        help="Q, K Parameter Quantization 4bit PACT")
    
    parser.add_argument('--loss_SM',
                        default =False, type=str2bool,
                        help="per layer loss coeffficient Softmax")
    
    parser.add_argument('--bert',
                        default ="base", type=str,
    )

    parser.add_argument('--act_method',
                        default ="clipping", type=str,
    )

    parser.add_argument('--step1_option',
                        default ="map", type=str,
    )

    parser.add_argument("--per_gpu_batch_size",
                        default=16,
                        type=int,
                        help="Per GPU batch size for training.")

    args = parser.parse_args() 
    
    # ================================================================================  #
    # Logging setup
    # ================================================================================ #
    run = None

    # Use Neptune for logging
    if args.neptune:
        import neptune.new as neptune
        # run = neptune.init(project='Neptune_ID/' + args.task_name.upper(),
        #             api_token='Neptune_API_Token')
        run = neptune.init(project='niceball0827/' + args.task_name.upper(),
                    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLC\
                    JhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YjM\
                    0ZTYwMi1kNjQwLTQ4NGYtOTYxMy03Mjc5ZmVkMzY2YTgifQ==')

    # ================================================================================  #
    # Load Directory
    # ================================================================================ #
    
    # Exp Name
    exp_name = args.exp_name 

    if args.loss_SM:
        exp_name = exp_name + "_" + str(args.sm_temp)
    if args.attn_distill:
        exp_name += "_S"
    if args.attnmap_distill:
        exp_name += "_M"
    if args.context_distill:
        exp_name += "_C"
    if args.output_distill:
        exp_name += "_O"
    args.exp_name = exp_name
    
    # Print Setting Info
    # logger.info(f"DISTILL1 => rep : {args.rep_distill} | cls : {args.pred_distill} | atts : {args.attn_distill} | attmap : {args.attnmap_distill} | tc_insert : {args.teacher_attnmap} | gt_loss : {args.gt_loss}")
    # logger.info(f"DISTILL2 => S : {args.attn_distill} M : {args.attnmap_distill} C: {args.context_distill}  O: {args.output_distill}")
    # logger.info(f"COEFF => attnmap : {args.attnmap_coeff} | attnmap_word : {args.word_coeff} | cls_coeff : {args.cls_coeff} | att_coeff : {args.att_coeff} | rep_coeff : {args.rep_coeff}")
    logger.info(f'EXP SET: {exp_name}')
    logger.info(f"SEED: {args.seed}")
    logger.info(f'EXP SET: {exp_name}')
    # logger.info(f"QK-FP : {args.qk_FP}")
    # logger.info(f"STOP-GRAD : {args.stop_grad}")
    # logger.info(f"Others LR : {args.other_lr}")

    
    if args.teacher_input:
        logger.info("EXP SET : TEACHER INPUT")

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
    
    if args.bert == "tiny":
        args.model_dir = os.path.join(args.model_dir, "BERT_Tiny")
        args.output_dir = os.path.join(args.output_dir, "BERT_Tiny")
    
    # Model Save Directory
    output_dir = os.path.join(args.output_dir,task_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # ================================================================================  #
    # Load Pths
    # ================================================================================ #
    # Student Model Pretrained FIle    
    
    if args.training_type == "downstream":
        args.student_model = os.path.join(args.model_dir, "BERT_base")
    elif args.training_type == "qat_normal":
        args.student_model = os.path.join(args.model_dir,task_name) 
        # args.student_model = os.path.join(args.output_dir, task_name, "exploration", "1SB_1epoch_S")
    elif args.training_type == "qat_step1":
        args.student_model = os.path.join(args.model_dir, task_name) 
    
    elif args.training_type == "qat_step2": 
        if args.step1_option == "map":
            args.student_model = os.path.join(args.output_dir, task_name, "exploration", "sarq_step1_TI_S_M")
        elif args.step1_option == "cc":
            args.student_model = os.path.join(args.output_dir, task_name, "exploration", "sarq_step1_ci_c_C")
            # args.student_model = os.path.join(args.output_dir, task_name, "quant", "sarq_step1_ci_c")
        elif args.step1_option == "co":
            args.student_model = os.path.join(args.output_dir, task_name, "exploration", "sarq_step1_CI_C_O")
        elif args.step1_option == "three":
            args.student_model = os.path.join(args.output_dir, task_name, "quant", "sarq_step1.5_ci_c")
            # args.student_model = os.path.join(args.output_dir, task_name, "last", "sarq_step1.5_ci_c_l")
        else:
            args.student_model = os.path.join(args.output_dir, task_name, "quant", "sarq_step1")
    
    elif args.training_type == "qat_step3":
        args.student_model = os.path.join("output", task_name, "quant", "step2_pact_4bit")
    elif args.training_type == "gradual": # For Gradual Quantization
        args.student_model = os.path.join("output", task_name, "quant", "2SB_4bit_save") 
    else:
        raise ValueError("Choose Training Type {downsteam, qat_normal, qat_step1, qat_step2, qat_step3, gradual}")

    # Teacher Model Pretrained FIle    
    args.teacher_model = os.path.join(args.model_dir,task_name)


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
    
    args.batch_size = n_gpu*args.per_gpu_batch_size

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=True)

    # ================================================================================  #
    # Dataset Setup (with DA)
    # ================================================================================ #

    input_file = 'train-v2.0' if args.version_2_with_negative else 'train-v1.1'
    input_file = os.path.join(args.data_dir,input_file)
    processed_data_dir = os.path.join(args.data_dir, 'processed')
    
    try:
        train_file = os.path.join(processed_data_dir, 'data.pkl')
        train_features = pickle.load(open(train_file,'rb'))
    except:
        input_file = 'train-v2.0.json' if args.version_2_with_negative else 'train-v1.1.json'
        input_file = os.path.join(args.data_dir,input_file)
        _, train_examples = read_squad_examples(
                        input_file=input_file, is_training=True,
                        version_2_with_negative=args.version_2_with_negative)
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True)
        with open(train_file, 'wb') as f:
                pickle.dump(train_features, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    num_train_optimization_steps = math.ceil(
        len(train_features) / args.batch_size) * args.num_train_epochs

    logger.info("***** Running training *****")
    logger.info("  Num split examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    # preparing dev data
    input_file = 'dev-v2.0.json' if args.version_2_with_negative else 'dev-v1.1.json'
    args.dev_file = os.path.join(args.data_dir,input_file)
    try:
        dev_file = os.path.join(processed_data_dir, "dev.pkl")
        eval_features = pickle.load(open(dev_file,'rb'))
        dev_dataset, eval_examples = read_squad_examples(
                            input_file=args.dev_file, is_training=False,
                            version_2_with_negative=args.version_2_with_negative)
    except:
        input_file = 'dev-v2.0.json' if args.version_2_with_negative else 'dev-v1.1.json'
        args.dev_file = os.path.join(args.data_dir,input_file)
        dev_dataset, eval_examples = read_squad_examples(
                            input_file=args.dev_file, is_training=False,
                            version_2_with_negative=args.version_2_with_negative)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)
        with open(dev_file, 'wb') as f:
                pickle.dump(eval_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    # ================================================================================ #
    # Build Teacher Model
    # ================================================================================ # 
    teacher_model = BertForQuestionAnswering.from_pretrained(args.teacher_model)
    teacher_model.to(device)
    teacher_model.eval()
    
    if n_gpu > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)
    
    result = do_eval(args,teacher_model, eval_dataloader,eval_features,eval_examples,args.device, dev_dataset)
    fp_em,fp_f1 = result['exact_match'],result['f1']
    logger.info(f"Full precision teacher exact_match={fp_em},f1={fp_f1}")
    
    # ================================================================================  #
    # Build Student Model
    # ================================================================================ #
    student_config = BertConfig.from_pretrained(args.student_model, 
                                                quantize_act=args.act_quant,
                                                quantize_weight=args.weight_quant,
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
                                                act_quantizer = args.act_quantizer,
                                                init_scaling = args.init_scaling,
                                                clip_ratio = args.clip_ratio,
                                                gradient_scaling = args.gradient_scaling,
                                                clip_method = args.clip_method,
                                                teacher_attnmap = args.teacher_attnmap,
                                                teacher_context = args.teacher_context,
                                                teacher_input = args.teacher_input,
                                                layer_thres_num= args.layer_thres_num,
                                                parks = args.parks,
                                                stop_grad = args.stop_grad,
                                                qk_FP = args.qk_FP,
                                                qkv_FP = args.qkv_FP,
                                                map=args.map,
                                                sm_temp=args.sm_temp,
                                                loss_SM=args.loss_SM,
                                                act_method = args.act_method
                                                )
    
    student_model = QuantBertForQuestionAnswering.from_pretrained(args.student_model, config = student_config, num_labels=num_labels)
    
    student_model.to(device)

    # ACT Quantization Option
    if args.act_quantizer != "ternary" and args.act_quant:
        
        for name, module in student_model.named_modules():
            if isinstance(module, (QuantizeLinear, QuantizeAct, ClipLinear)):    
                module.act_flag = False
                module.weight_flag = False
        
        # student_model.eval()
        # result = do_eval(student_model, task_name, eval_dataloader,
        #                             device, output_mode, eval_labels, num_labels, teacher_model=teacher_model)
        # print(result)
        
        cv_initialize(student_model, train_dataloader, torch.Tensor([args.index_ratio]), device)    
        
        for name, module in student_model.named_modules():
            if isinstance(module, (QuantizeLinear, QuantizeAct, ClipLinear)):
                module.act_flag = args.act_quant
                module.weight_flag = args.weight_quant      
    
    for name, module in student_model.named_modules():
        if isinstance(module, ClipLinear):
            module.act_flag = args.act_quant
            module.weight_flag = args.weight_quant      
    
    # ================================================================================  #
    # Training Setting
    # ================================================================================ #
    if n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)
    param_optimizer = list(student_model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'temperature']
    clip_decay = ['clip_val', 'clip_valn']
    coeff = ["coeff"]

    pact_quantizer = args.quantizer == "pact" or args.act_quantizer == "pact"
    
    if args.task_name == "cola" and args.training_type == "qat_step2":
        args.learning_rate = 1E-4

    {'params': [p for n, p in param_optimizer if any(nd in n for nd in coeff)], 'weight_decay': 0.0, 'lr':args.learning_rate * 50},
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in (no_decay+clip_decay+coeff))], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(cv in n for cv in clip_decay)],\
        'weight_decay': args.clip_wd if pact_quantizer else 0, 'lr': args.learning_rate * args.lr_scaling}
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
    l_loss = AverageMeter()
    
    # grad_dict = dict()
    # for l in range(student_config.num_hidden_layers):
    #     grad_dict[f"ffn_1_L{l}"] = []
    #     grad_dict[f"ffn_2_L{l}"] = []
    #     grad_dict[f"query_L{l}"] = []
    #     grad_dict[f"key_L{l}"] = []
    #     grad_dict[f"value_L{l}"] = []

    #layer_attmap_loss = [ AverageMeter() for i in range(12) ]
    #layer_att_loss = [ AverageMeter() for i in range(12) ]
    #layer_rep_loss = [ AverageMeter() for i in range(13) ] # 12 Layers Representation, 1 Word Embedding Layer 

    for epoch_ in range(int(args.num_train_epochs)):
        # logger.info("****************************** %d Epoch ******************************", epoch_)
        nb_tr_examples, nb_tr_steps = 0, 0
        student_config.layer_thres_num += 6*epoch_

        for batch in tqdm(train_dataloader,desc=f"Epoch_{epoch_}", mininterval=0.01, ascii=True):
            
            student_model.train()
            
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
            
            # By Summing input mask, We can get Sequence Length
            seq_lengths = []
            for b in range(input_mask.shape[0]):
                seq_lengths.append(input_mask[b].sum().item())
            seq_lengths = torch.Tensor(seq_lengths).to(args.device)

            # tmp loss init
            att_loss = 0.
            attmap_loss = 0.
            rep_loss = 0.
            cls_loss = 0.
            attscore_loss = 0.
            output_loss = 0.
            loss = 0.
            
            with torch.no_grad():
                teacher_logits, teacher_atts, teacher_reps, teacher_probs, teacher_values = teacher_model(input_ids, segment_ids, input_mask)
        
            student_logits, student_atts, student_reps, student_probs, student_values = student_model(input_ids, segment_ids, input_mask, teacher_outputs=(teacher_probs, teacher_values, teacher_reps, teacher_logits), output_mode=output_mode, seq_lengths=seq_lengths)
            # _, loss, cls_loss, rep_loss, output_loss, attmap_loss, attscore_loss, student_zip  = student_model(input_ids, segment_ids, input_mask, teacher_outputs=(teacher_probs, teacher_values, teacher_reps, teacher_logits, teacher_atts), output_mode=output_mode, seq_lengths=seq_lengths)

            # Pred Loss
            if args.pred_distill:
                soft_start_ce_loss = soft_cross_entropy(student_logits[0], teacher_logits[0])
                soft_end_ce_loss = soft_cross_entropy(student_logits[1], teacher_logits[1])

                cls_loss = soft_start_ce_loss + soft_end_ce_loss
                l_cls_loss.update(cls_loss.item())
            
            # Output Loss
            if args.output_distill:
                for i, (student_value, teacher_value) in enumerate(zip(student_values, teacher_values)):    
                    tmp_loss = MSELoss()(student_value[1], teacher_value[1]) # 1 : Attention Output 0 : Layer Context
                    output_loss += tmp_loss
                l_output_loss.update(output_loss.item())
            
            # Attention Score Loss
            if args.attn_distill:
                for i, (student_att, teacher_att) in enumerate(zip(student_atts, teacher_atts)):    
                            
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to("cuda"),
                                                student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to("cuda"),
                                                teacher_att)

                    tmp_loss = MSELoss()(student_att, teacher_att)
                    attscore_loss += tmp_loss
                l_att_loss.update(attscore_loss.item())

            # Attention Map Loss
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
                            
                    # student_prob = student_prob["attn"]
                    # teacher_prob = teacher_prob["attn"]

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
                    
                    # kld_loss_sum = torch.sum(kld_loss)
                    kld_loss_mean = torch.mean(kld_loss)

                    # Other Option (Cosine Similarity, MSE Loss)
                    # attnmap_mse_loss = loss_mse(student, teacher)
                    #kld_loss_sum = torch.nn.functional.cosine_similarity(student, teacher, -1).mean()
                    

                    #layer_attmap_loss[i].update(kld_loss_sum)
                    # attmap_loss += attnmap_mse_loss
                    attmap_loss += kld_loss_mean
                
                l_attmap_loss.update(attmap_loss.item())
            
            # Rep Distill
            if args.rep_distill:
                for i, (student_rep, teacher_rep) in enumerate(zip(student_reps, teacher_reps)):
                    tmp_loss = MSELoss()(student_rep, teacher_rep)
                    rep_loss += tmp_loss
                l_rep_loss.update(rep_loss.item())

            loss += cls_loss + rep_loss + attmap_loss + output_loss + attscore_loss
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
                    if task_name=='cola':
                        if run is not None:
                            run["metrics/mcc"].log(value=result['mcc'], step=global_step)
                    # logger.info(f"Eval Result is {result['mcc']}")
                    elif task_name in ['sst-2','mnli','mnli-mm','qnli','rte','wnli']:
                        if run is not None:
                            run["metrics/acc"].log(value=result['acc'],step=global_step)
                    # logger.info(f"Eval Result is {result['acc']}")
                    elif task_name in ['mrpc','qqp']:
                        if run is not None:
                            run["metrics/acc_and_f1"].log(value=result['acc_and_f1'],step=global_step)
                        # logger.info(f"Eval Result is {result['acc']}, {result['f1']}")
                    else:
                        if run is not None:
                            run["metrics/corr"].log(value=result['corr'],step=global_step)
 
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()
            
            global_step += 1
            save_model = False
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
                
                result = do_eval(args,student_model, eval_dataloader,eval_features,eval_examples,args.device, dev_dataset, teacher_model=teacher_model)
                em,f1 = result['exact_match'],result['f1']
                logger.info(f'FP {fp_em}/{fp_f1}')
                logger.info(f'{em}/{f1}')

                run["metrics/acc_em"].log(value=em, step=global_step)
                run["metrics/acc_f1"].log(value=f1, step=global_step)
                run["metrics/acc_em and f1"].log(value=(f1+em)/2, step=global_step)

                result['global_step'] = global_step
                result['cls_loss'] = l_cls_loss.avg
                result['att_loss'] = l_att_loss.avg
                result['rep_loss'] = l_rep_loss.avg
                result['loss'] = l_loss.avg
                
                # Basic Logging (Training Loss, Clip Val)
                if run is not None:
                    
                    run["loss/total_loss"].log(value=l_loss.avg, step=global_step)
                    run["loss/gt_loss_loss"].log(value=l_gt_loss.avg, step=global_step)
                    run["loss/att_loss_loss"].log(value=l_att_loss.avg, step=global_step)
                    run["loss/rep_loss_loss"].log(value=l_rep_loss.avg, step=global_step)
                    run["loss/cls_loss_loss"].log(value=l_cls_loss.avg, step=global_step)
                    run["loss/output_loss_loss"].log(value=l_output_loss.avg, step=global_step)
                    run["loss/attmap_loss_loss"].log(value=l_attmap_loss.avg, step=global_step)
                    
                    run["metrics/lr"].log(value=optimizer.get_lr()[0], step=global_step)

                # Save Model
            
            if save_model:
                logger.info(previous_best)
                if args.save_fp_model:
                    logger.info("******************** Save full precision model ********************")
                    model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(args.output_dir)
                if args.save_quantized_model:
                    logger.info("******************** Save quantized model ********************")
                    output_quant_dir = os.path.join(args.output_dir, 'quant')
                    if not os.path.exists(output_quant_dir):
                        os.makedirs(output_quant_dir)
                    
                    output_quant_dir = os.path.join(output_quant_dir, args.exp_name)
                    if not os.path.exists(output_quant_dir):
                        os.makedirs(output_quant_dir)
                        
                    model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                    quant_model = copy.deepcopy(model_to_save)
                    for name, module in quant_model.named_modules():
                        if hasattr(module,'weight_quantizer'):
                            module.weight.data = module.weight_quantizer.apply(module.weight,module.weight_clip_val,
                                                                         module.weight_bits,True)

                    output_model_file = os.path.join(output_quant_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(output_quant_dir, CONFIG_NAME)

                    torch.save(quant_model.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(output_quant_dir)
                

    

    logger.info(f"==> Previous Best = {previous_best}")
    logger.info(f"==> Last Result = {result}")


if __name__ == "__main__":
    main()
