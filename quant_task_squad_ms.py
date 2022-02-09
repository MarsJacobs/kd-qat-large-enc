from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
import pickle
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MSELoss

from transformer import BertForQuestionAnswering,WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling_quant import BertForQuestionAnswering as QuantBertForQuestionAnswering
from transformer import BertTokenizer
from transformer import BertAdam
from transformer import BertConfig

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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return torch.sum((- targets_prob * student_likelihood), dim=-1).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default='data/',
                        type=str,
                        help="The data directory.")
    parser.add_argument("--model_dir",
                        default='models/',
                        type=str,
                        help="The models directory.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The models directory.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        help="The models directory.")
    parser.add_argument("--output_dir",
                        default='output',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--version_2_with_negative',
                        action='store_true', 
                        help="Squadv2.0 if true else Squadv1.1 ")

    # default
    parser.add_argument("--max_seq_length",
                        default=384,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--doc_stride", 
                        default=128, 
                        type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", 
                        default=64, 
                        type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--n_best_size", 
                        default=20, 
                        type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", 
                        default=30, 
                        type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", 
                        default=0, 
                        type=int)
    parser.add_argument('--null_score_diff_threshold',
                        type=float, 
                        default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_lower_case',
                        #action='store_true',
                        default=True,
                        help="do lower case")
    

    parser.add_argument("--per_gpu_batch_size",
                        default=16,
                        type=int,
                        help="Per GPU batch size for training.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--eval_step', 
                        type=int, 
                        default=3000,
                        help="Evaluate every X training steps")

    parser.add_argument("--task_name",
                        default='SQUADv2',
                        type=str,
                        help="The name of the task to train.")

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
    parser.add_argument("--clip_val",
                        default=2.5,
                        type=float,
                        help="Initial clip value.")
    
    parser.add_argument('--log_map',
                        default=False, type=str2bool,
                        )
    
    parser.add_argument('--log_metric',
                        default=False, type=str2bool,
                        )

    parser.add_argument("--tc_top_k",
                        default=3,
                        type=int,
                        help="Top-K Coverage")

    parser.add_argument("--gpus",
                        default=1,
                        type=int,
                        help="Number of GPUs to use")

    parser.add_argument('--act_quant',
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
    
    parser.add_argument("--other_lr",
                        default=0.0001,
                        type=float,
                        help="other param lr")

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
    
    parser.add_argument('--gt_loss',
                        default =True, type=str2bool,
                        help="Ground Truth Option")
    
    parser.add_argument('--value_relation',
                        default =False, type=str2bool,
                        help="attention Map Distill Option")

    parser.add_argument('--teacher_attnmap',
                        default =False, type=str2bool,
                        help="attention Map Distill Option")
    
    parser.add_argument('--map',
                        default =False, type=str2bool,
                        help="Q, K Parameter Quantization 4bit PACT")
    args = parser.parse_args()

    run = None
    # Use Neptune for logging
    if args.neptune:
        import neptune.new as neptune
        run = neptune.init(project='niceball0827/' + args.task_name,
                    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLC\
                    JhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YjM\
                    0ZTYwMi1kNjQwLTQ4NGYtOTYxMy03Mjc5ZmVkMzY2YTgifQ==')
    
    logger.info("SQUAD Dataset")
    logger.info(f"DISTILL => rep : {args.rep_distill} | cls : {args.pred_distill} | atts : {args.attn_distill} | attmap : {args.attnmap_distill} | tc_insert : {args.teacher_attnmap} | gt_loss : {args.gt_loss}")
    logger.info(f"COEFF => attnmap : {args.attnmap_coeff} | cls_coeff : {args.cls_coeff} | att_coeff : {args.att_coeff} | rep_coeff : {args.rep_coeff}")
    logger.info('The args: {}'.format(args))

    # ================================================================================  #
    # Load Pths
    # ================================================================================ #
    if args.teacher_model is None:
        if args.version_2_with_negative:
            args.teacher_model = os.path.join("models", "squadv2.0")
        else:
            args.teacher_model = os.path.join("models", "squadv1.1")
    if args.student_model is None:
        if args.version_2_with_negative:
            args.student_model = os.path.join("models", "squadv2.0")
        else:
            args.student_model = os.path.join("models", "squadv1.1")

    # ================================================================================  #
    # prepare devices & seed
    # ================================================================================ #
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    args.batch_size = args.n_gpu*args.per_gpu_batch_size

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=True)

    # ================================================================================  #
    # Dataset Setup 
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
        
    num_train_optimization_steps = int(
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


    # ================================================================================  #
    # Build Teacher Model
    # ================================================================================ # 

    teacher_model = BertForQuestionAnswering.from_pretrained(args.teacher_model)
    teacher_model.to(args.device)
    teacher_model.eval()
    if args.n_gpu > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)
    result = do_eval(args,teacher_model, eval_dataloader,eval_features,eval_examples,args.device, dev_dataset)
    fp_em,fp_f1 = result['exact_match'],result['f1']
    logger.info(f"Full precision teacher exact_match={fp_em},f1={fp_f1}")
    
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
                                                teacher_attnmap = args.teacher_attnmap,
                                                parks = args.parks,
                                                stop_grad = args.stop_grad,
                                                qk_FP = args.qk_FP,
                                                map=args.map
                                                )

    student_model = QuantBertForQuestionAnswering.from_pretrained(args.student_model, config = student_config)
    
    for name, module in student_model.named_modules():
        if hasattr(module,'weight_quantizer'):
            try:
                module.clip_initialize()
            except:
                import pdb; pdb.set_trace()
    
    student_model.to(args.device)

    # ================================================================================ #
    # Training Setting
    # ================================================================================ #

    if args.n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)
    
    # Prepare optimizer    
    if args.training_type == "qat_step1" or args.training_type == "qat_step2" or args.training_type == "qat_step3":

        train_param_list = []
        freeze_param_list = []
        qk_no_decay_list = []
        no_decay_list = []
    

        for n, p in student_model.named_parameters():
            if "self.query.weight" in n or "self.key.weight" in n or "self.query.bias" in n or "self.key.bias" in n:
                if "self.query.weight" in n or "self.key.weight" in n:
                    train_param_list.append(p)
                if "self.query.bias" in n or "self.key.bias" in n:
                    qk_no_decay_list.append(p)
            else:
                if "bias" in n or "LayerNorm.bias" in n or "layerNorm.weight" in n:
                    no_decay_list.append(p)
                else:
                    freeze_param_list.append(p)
                #p.requires_grad = False
        
        if args.training_type == "qat_step1": # main : other param
            optimizer_grouped_parameters = [
                {'params': train_param_list, 'weight_decay': 0.01, 'lr' : args.other_lr},
                {'params': qk_no_decay_list, 'weight_decay': 0.0, 'lr' : args.other_lr},
                {'params': freeze_param_list,'weight_decay': 0.01, 'lr': args.learning_rate},
                {'params': no_decay_list,'weight_decay': 0.0, 'lr': args.learning_rate},
            ]

        if args.training_type == "qat_step2" or args.training_type == "qat_step3":
            optimizer_grouped_parameters = [
                {'params': train_param_list, 'weight_decay': 0.01, 'lr' : args.learning_rate},
                {'params': qk_no_decay_list, 'weight_decay': 0.0, 'lr' : args.learning_rate},
                {'params': freeze_param_list,'weight_decay': 0.01, 'lr': args.other_lr},
                {'params': no_decay_list,'weight_decay': 0.0, 'lr': args.other_lr},
            ]


    elif args.training_type == "qat_normal" or args.training_type == "downstream" or args.training_type == "gradual":
        param_optimizer = list(student_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        clip_decay = ['clip_val', 'clip_valn']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in (no_decay+clip_decay))], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer if any(cv in n for cv in clip_decay)],\
            'weight_decay': args.clip_wd if args.quantizer == "pact" else 0, 'lr': args.learning_rate * args.lr_scaling}
        ]
    
    else:
         raise ValueError("Choose Training Type {downsteam, qat_normal, qat_step1, qat_step2}")
    schedule = 'warmup_linear'
    optimizer = BertAdam(optimizer_grouped_parameters,
                            schedule=schedule,
                            lr=args.learning_rate,
                            warmup=0.1,
                            t_total=num_train_optimization_steps)

    loss_mse = MSELoss()
    # Train and evaluate

    global_step = 0
    best_dev_f1 = 0.0
    flag_loss = float('inf')
    previous_best = None
    
    tr_loss = 0.
    tr_att_loss = 0.
    tr_rep_loss = 0.
    tr_cls_loss = 0.

    # Loss Init AverageMeter
    l_gt_loss = AverageMeter()
    l_attmap_loss = AverageMeter()
    l_att_loss = AverageMeter()
    l_rep_loss = AverageMeter()
    l_cls_loss = AverageMeter()
    l_loss = AverageMeter()
    
    layer_attmap_loss = [ AverageMeter() for i in range(12) ]
    layer_att_loss = [ AverageMeter() for i in range(12) ]
    layer_rep_loss = [ AverageMeter() for i in range(13) ] # 12 Layers Representation, 1 Word Embedding Layer 

    for epoch_ in range(int(args.num_train_epochs)):
        logger.info("****************************** %d Epoch ******************************", epoch_)

        for step, batch in enumerate(train_dataloader):
            student_model.train()
            batch = tuple(t.to(args.device) for t in batch)

            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
            
            att_loss = 0.
            attmap_loss = 0.
            rep_loss = 0.
            cls_loss = 0.
            loss = 0.

            student_logits, student_atts, student_reps, student_probs, student_values = student_model(input_ids,segment_ids,input_mask)
            with torch.no_grad():
                teacher_logits, teacher_atts, teacher_reps, teacher_probs, teacher_values = teacher_model(input_ids, segment_ids, input_mask)

            if args.pred_distill:
                soft_start_ce_loss = soft_cross_entropy(student_logits[0], teacher_logits[0])
                soft_end_ce_loss = soft_cross_entropy(student_logits[1], teacher_logits[1])
                cls_loss = soft_start_ce_loss + soft_end_ce_loss

                l_cls_loss.update(cls_loss.item())
                cls_loss = cls_loss * args.cls_coeff
            
            if args.attnmap_distill:

                BATCH_SIZE = student_probs[0].shape[0]
                NUM_HEADS = student_probs[0].shape[1]
                MAX_SEQ = student_probs[0].shape[2]
                
                mask = torch.zeros(BATCH_SIZE, NUM_HEADS, MAX_SEQ, MAX_SEQ, dtype=torch.float32)
                mask_seq = []
                
                for sent in range(BATCH_SIZE):
                    s = seq_lengths[sent]
                    mask[sent, :, :s, :s] = 1.0
                
                mask = mask.to(args.device)

                for i, (student_prob, teacher_prob) in enumerate(zip(student_probs, teacher_probs)):            

                    # KLD(teacher || student)
                    # = sum (p(t) log p(t)) - sum(p(t) log p(s))
                    # = (-entropy) - (-cross_entropy)
                    
                    student = torch.clamp_min(student_prob, 1e-8)
                    teacher = torch.clamp_min(teacher_prob, 1e-8)
                    
                    # Other Option (Cosine Similarity, MSE Loss)

                    #kld_loss_sum = loss_mse(student, teacher)
                    #kld_loss_sum = torch.nn.functional.cosine_similarity(student, teacher, -1).mean()
                    
                    # p(t) log p(s) = negative cross entropy
                    neg_cross_entropy = teacher * torch.log(student) * mask
                    neg_cross_entropy = torch.sum(neg_cross_entropy, dim=-1)  # (b, h, s, s) -> (b, h, s)
                    neg_cross_entropy = torch.sum(neg_cross_entropy, dim=-1) / seq_lengths.view(-1, 1)  # (b, h, s) -> (b, h)

                    # p(t) log p(t) = negative entropy
                    neg_entropy = teacher * torch.log(teacher) * mask
                    neg_entropy = torch.sum(neg_entropy, dim=-1)  # (b, h, s, s) -> (b, h, s)
                    neg_entropy = torch.sum(neg_entropy, dim=-1) / seq_lengths.view(-1, 1)  # (b, h, s) -> (b, h)

                    kld_loss = neg_entropy - neg_cross_entropy
                    
                    kld_loss_sum = torch.sum(kld_loss)

                    layer_attmap_loss[i].update(kld_loss_sum)
                    attmap_loss += kld_loss_sum
                
                l_attmap_loss.update(attmap_loss.item())
                attmap_loss = args.attnmap_coeff*attmap_loss

            if args.attn_distill:

                for i, (student_att, teacher_att) in enumerate(zip(student_atts, teacher_atts)):    
                    
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(args.device),
                                                student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(args.device),
                                                teacher_att)

                    if args.kd_layer_num != -1:
                        if i == args.kd_layer_num:
                            tmp_loss = loss_mse(student_att, teacher_att)
                        else:   
                            tmp_loss = loss_mse(student_att, teacher_att)
                            tmp_loss = tmp_loss * 0
                    else:
                        tmp_loss = loss_mse(student_att, teacher_att)
                    
                    layer_att_loss[i].update(tmp_loss)
                    att_loss += tmp_loss 
                    
                l_att_loss.update(att_loss.item())
                att_loss = args.att_coeff * att_loss

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
                   
                    layer_rep_loss[i].update(tmp_loss)
                    rep_loss += tmp_loss

                l_rep_loss.update(rep_loss.item())
                rep_loss = args.rep_coeff * rep_loss
                

            loss += cls_loss + rep_loss + attmap_loss + att_loss
            l_loss.update(loss.item())
            
            # Zero Step Loss Update
            if global_step == 0: 
                if run is not None:           
                    run["loss/total_loss"].log(value=l_loss.avg, step=global_step)
                    run["loss/gt_loss_loss"].log(value=l_gt_loss.avg, step=global_step)
                    run["loss/att_loss_loss"].log(value=l_att_loss.avg, step=global_step)
                    run["loss/rep_loss_loss"].log(value=l_rep_loss.avg, step=global_step)
                    run["loss/cls_loss_loss"].log(value=l_cls_loss.avg, step=global_step)
                    run["loss/attmap_loss_loss"].log(value=l_attmap_loss.avg, step=global_step)
                    run["metrics/lr"].log(value=optimizer.get_lr()[0], step=global_step)

                    for i in range(12):
                        
                        run[f"loss/layer_{i}_att_loss_loss"].log(value=layer_att_loss[i].avg, step=global_step)
                        run[f"loss/layer_{i}_rep_loss_loss"].log(value=layer_rep_loss[i].avg, step=global_step)
                        run[f"loss/layer_{i}_attmap_loss_loss"].log(value=layer_attmap_loss[i].avg, step=global_step)
            
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            save_model = False
            # ================================================================================  #
            #  Evaluation
            # ================================================================================ #
            if global_step % args.eval_step == 0 or global_step == num_train_optimization_steps-1:
                logger.info("***** Running evaluation *****")
                logger.info(f"  Epoch = {epoch_} iter {global_step} step")
                if previous_best is not None:
                    logger.info(f"Previous best = {previous_best}")

                student_model.eval()
                result = do_eval(args,student_model, eval_dataloader,eval_features,eval_examples,args.device, dev_dataset)
                em,f1 = result['exact_match'],result['f1']
                logger.info(f'FP {fp_em}/{fp_f1}')
                logger.info(f'{em}/{f1}')
                if f1 > best_dev_f1:
                    previous_best = f"exact_match={em},f1={f1}"
                    best_dev_f1 = f1
                    save_model = True

                # Basic Logging (Training Loss)
                if run is not None:
                    
                    run["loss/total_loss"].log(value=l_loss.avg, step=global_step)
                    run["loss/gt_loss_loss"].log(value=l_gt_loss.avg, step=global_step)
                    run["loss/att_loss_loss"].log(value=l_att_loss.avg, step=global_step)
                    run["loss/rep_loss_loss"].log(value=l_rep_loss.avg, step=global_step)
                    run["loss/cls_loss_loss"].log(value=l_cls_loss.avg, step=global_step)
                    run["loss/attmap_loss_loss"].log(value=l_attmap_loss.avg, step=global_step)
                    run["metrics/lr"].log(value=optimizer.get_lr()[0], step=global_step)

                    for i in range(12):

                        run[f"loss/layer_{i}_att_loss_loss"].log(value=layer_att_loss[i].avg, step=global_step)
                        run[f"loss/layer_{i}_rep_loss_loss"].log(value=layer_rep_loss[i].avg, step=global_step)
                        run[f"loss/layer_{i}_attmap_loss_loss"].log(value=layer_attmap_loss[i].avg, step=global_step)
               

                
            #save quantiozed model
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


if __name__ == "__main__":
    main()
