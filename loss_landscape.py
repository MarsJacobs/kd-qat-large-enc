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
from transformer import QuantizeLinear, QuantizeAct, BertSelfAttention, FP_BertSelfAttention, ClipLinear, BertAttention, FP_BertAttention
from utils_glue import *
from bertviz import model_view

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn.functional as F

import ops.tests as tests
import ops.datasets as datasets
import ops.loss_landscapes as lls

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


def main():
    # ================================================================================  #
    # ArgParse
    # ================================================================================ #
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name",
                        default='sst-2',
                        type=str,
                        help="The name of the task to train.")

    parser.add_argument("--model_name",
                        default='sst-2',
                        type=str,
                        help="The name of the task to train.")

    args = parser.parse_args() 

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

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = "models"
    output_dir = "output"
    bert_size = "base"

    if bert_size == "large":
        model_dir = os.path.join(model_dir, "BERT_large")
        output_dir = os.path.join(output_dir, "BERT_large")

    teacher_model_dir = os.path.join(model_dir,args.task_name)

    # Processor & Task Info
    processor = processors[args.task_name]()
    output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

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
    student_model_dir = os.path.join(output_dir, args.task_name, "exploration", args.model_name)   
    student_config = BertConfig.from_pretrained(student_model_dir)   
    student_model = QuantBertForSequenceClassification.from_pretrained(student_model_dir, config = student_config, num_labels=num_labels)
    student_model.to(device)

    scale = 1e-0
    n = 21
    gpu = torch.cuda.is_available()

    metrics_grid = lls.get_loss_landscape(
        student_model, 1, train_dataloader, transform=None,
        kws=["pos_embed", "relative_position"],
        x_min=-1.0 * scale, x_max=1.0 * scale, n_x=n, y_min=-1.0 * scale, y_max=1.0 * scale, n_y=n, gpu=gpu
    )

    metrics_dir = os.path.join("lls_logs", "%s_%s_lls.csv" % (args.task_name, args.model_name))
    metrics_list = [[*grid, metrics] for grid, metrics in metrics_grid.items()]

    with open(metrics_dir, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for metrics in metrics_list:
            writer.writerow(metrics)

if __name__ == "__main__":
    main()