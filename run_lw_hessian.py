
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

from tqdm import tqdm
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
from transformer import QuantizeLinear, QuantizeAct, BertSelfAttention, FP_BertSelfAttention, ClipLinear, BertAttention, FP_BertAttention
from utils_glue import *
from bertviz import model_view

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return torch.sum((- targets_prob * student_likelihood), dim=-1).mean()


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

def group_product(xs, ys):
        """
        the inner product of two lists of variables xs,ys
        :param xs:
        :param ys:
        :return:
        """
        return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def de_variable(v):
    '''
    normalize the vector and detach it from variable
    '''

    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item() + 1e-6
    v = [vi / s for vi in v]
    return v

def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    # v = [vi / s for vi in v]
    return v


def orthonormal(w, v_list):
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)


def total_number_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name",
                        default="cola",
                        type=str
                        )   

    parser.add_argument("--model_name",
                        default="1SB_M",
                        type=str
                        )

    parser.add_argument("--bert_size",
                        default='base',
                        type=str
                        )
    
    parser.add_argument("--data_percentage",
                        default=0.01,
                        type=float,
                        )

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    
    parser.add_argument('--tol',
                        type=float,
                        default=0.01,
                        help="random seed for initialization")
    
    parser.add_argument('--kd_loss',
                        default=False, 
                        type=str2bool,
                        )

    parser.add_argument('--kd_loss_type', 
                        type=str,
                        )
    

    args = parser.parse_args() 

    logger.info(f"SEED: {args.seed}")
    logger.info(f"TOL: {args.tol}")
    logger.info(f"DATA: {args.data_percentage}")

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
            "cola": {"max_seq_length": 64,"batch_size":16,"eval_step": 400}, # No Aug : 50 Aug : 400
            "mnli": {"max_seq_length": 128,"batch_size":32,"eval_step":8000},
            "mrpc": {"max_seq_length": 128,"batch_size":32,"eval_step":20},
            "sst-2": {"max_seq_length": 64,"batch_size":32,"eval_step":100},
            "sts-b": {"max_seq_length": 128,"batch_size":32,"eval_step":100},
            "qqp": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
            "qnli": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
            "rte": {"max_seq_length": 128,"batch_size":32,"eval_step":100}
        }

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = "models"
    output_dir = "output"
    bert_size = args.bert_size

    if bert_size == "large":
        model_dir = os.path.join(model_dir, "BERT_large")
        output_dir = os.path.join(output_dir, "BERT_large")

    teacher_model_dir = os.path.join(model_dir,args.task_name)

    # Processor & Task Info
    processor = processors[args.task_name]()
    output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args.bert_size == "large":
        layer_num = 24
        head_num = 16
    else: 
        layer_num = 12
        head_num = 12

    if args.task_name in default_params:
        batch_size = default_params[args.task_name]["batch_size"]
        max_seq_length = default_params[args.task_name]["max_seq_length"]
        eval_step = default_params[args.task_name]["eval_step"]
        
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(teacher_model_dir, do_lower_case=True)

    # Load Dataset
    data_dir = os.path.join("data",args.task_name)
    processed_data_dir = os.path.join(data_dir,'preprocessed')

    train_examples = processor.get_train_examples(data_dir)
    train_features = convert_examples_to_features(train_examples, label_list,
                                    max_seq_length, tokenizer, output_mode)

    train_data, train_labels = get_tensor_data(output_mode, train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Build Model
    if args.kd_loss:
        teacher_model = BertForSequenceClassification.from_pretrained(teacher_model_dir, num_labels=num_labels)
        teacher_model.to(device)
        teacher_model.eval()
    else:
        teacher_model = None

    student_model_dir = os.path.join(output_dir, args.task_name, "exploration", args.model_name)   
    student_config = BertConfig.from_pretrained(student_model_dir)   
    student_model = QuantBertForSequenceClassification.from_pretrained(student_model_dir, config = student_config, num_labels=num_labels)
    student_model.to(device)
    student_model.eval()

    percentage_index = len(train_dataloader.dataset) * args.data_percentage / batch_size
    print(f'percentage_index: {percentage_index}')

    # CSV File
    if args.kd_loss:
        csv_path = os.path.join("layer_hessian_results", f"{args.task_name}-{args.data_percentage}-{args.seed}-KD-{args.kd_loss_type}-{args.bert_size}-eigens.csv")
    else:
        csv_path = os.path.join("layer_hessian_results", f"{args.task_name}-{args.data_percentage}-{args.seed}-{args.bert_size}-eigens.csv")

    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['block', 'iters', 'max_eigenvalue'])

    # Layer Wise Hessian
    for module in student_model.modules():
        for param in module.parameters():
            param.requires_grad = True

    block_id = 0

    for block_id in tqdm(range(layer_num)):
        logger.info(f'block_id: {block_id}')
        model_block = student_model.bert.encoder.layer[block_id]
        
        v = [
                torch.randn(p.size()).to(device) for p in model_block.parameters()
            ]
        v = de_variable(v)

        lambda_old, lambdas = 0., 1.

        i = 0
        while (abs((lambdas - lambda_old) / lambdas) >= args.tol):

            lambda_old = lambdas

            acc_Hv = [
                torch.zeros(p.size()).cuda() for p in model_block.parameters()
            ]

            for step, batch in enumerate(train_dataloader):
                if step < percentage_index:
                    
                    tmp_loss = 0.
                    loss = 0.

                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids, _ = batch
                    
                    if args.kd_loss:
                        with torch.no_grad():
                            teacher_logits, teacher_atts, teacher_reps, teacher_probs, teacher_values = teacher_model(input_ids, segment_ids, input_mask)
        
                    student_logits, student_atts, student_reps, student_probs, student_values = student_model(input_ids, segment_ids, input_mask, teacher_outputs=None)

                    if args.kd_loss:
                        if args.kd_loss_type == "pred":
                            if output_mode == "classification":
                                loss = soft_cross_entropy(student_logits,teacher_logits)
                            elif output_mode == "regression":
                                loss = MSELoss()(student_logits, teacher_logits)
                            else:
                                loss = soft_cross_entropy(student_logits,teacher_logits)

                        elif args.kd_loss_type == "layer":
                            for i, (student_rep, teacher_rep) in enumerate(zip(student_reps, teacher_reps)):
                                tmp_loss = MSELoss()(student_rep, teacher_rep)
                                loss += tmp_loss
 
                    else:
                        if output_mode == "classification":
                            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                            loss = torch.nn.functional.nll_loss(lprobs, label_ids, reduction='sum')
                        elif output_mode == "regression":
                            loss = loss_mse(student_logits, teacher_logits)

                    loss.backward(create_graph=True)
                    grads = [param.grad for param in model_block.parameters()]
                    params = model_block.parameters()

                    Hv = torch.autograd.grad(
                        grads,
                        params,
                        grad_outputs=v,
                        only_inputs=True,
                        retain_graph=True)
                    acc_Hv = [
                        acc_Hv_p + Hv_p for acc_Hv_p, Hv_p in zip(acc_Hv, Hv)
                    ]
                    student_model.zero_grad()

            # calculate raylay quotients
            lambdas = group_product(acc_Hv, v).item() / percentage_index
            # logger.info(f'block_{block_id}-lambda: {lambdas}')
            v = de_variable(acc_Hv)

            if abs((lambdas - lambda_old) / lambdas) < args.tol:
                logger.info(f'==> CONVERGE block_{block_id}-lambda: {lambdas}')
                writer.writerow([f'{block_id}', f'{i}', f'{lambdas}'])
                csv_file.flush()

            i += 1
        


        






if __name__ == "__main__":
    main()