from __future__ import absolute_import, division, print_function

import pprint
import argparse
import logging
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3" # Set GPU Index to use
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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
# from torch.utils.tensorboard import SummaryWriter

from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from transformer import BertForSequenceClassification,WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling_quant import BertForSequenceClassification as QuantBertForSequenceClassification
from transformer import BertTokenizer
from transformer import BertAdam
from transformer import BertConfig
from transformer import QuantizeLinear, QuantizeAct, BertSelfAttention, FP_BertSelfAttention, ClipLinear
from utils_glue import *
from bertviz import model_view

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn.functional as F
        
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
        "cola": {"max_seq_length": 64,"batch_size":16,"eval_step":200}, #50
        "mnli": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "mrpc": {"max_seq_length": 128,"batch_size":32,"eval_step":200},
        "sst-2": {"max_seq_length": 64,"batch_size":32,"eval_step":200}, #64
        "sts-b": {"max_seq_length": 128,"batch_size":32,"eval_step":50},
        "qqp": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "qnli": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "rte": {"max_seq_length": 128,"batch_size":16,"eval_step":5000} # 100
    }

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        default='cola',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert",
                        default='models',
                        type=str,
                        )
    parser.add_argument("--file_name",
                        default='file',
                        type=str,
                        )
    parser.add_argument("--model_name",
                        default='file',
                        type=str,
                        )
    parser.add_argument("--quant_model_name",
                        default='file',
                        type=str,
                        )

    parser.add_argument('--init',
                        default =True, type=str2bool,
                        help="Whether to quantize Classifier Dense Layer")

    parser.add_argument('--kd_loss',
                        default =True, type=str2bool,
                        )

    parser.add_argument('--kd_loss_type',
                        default ="pred", type=str,
                        )
    
    parser.add_argument('--sample_N',
                        default =1, type=float
                        )

    
    args = parser.parse_args() 

    task_name = args.task
    bert_size = args.bert

    if bert_size == "large":
        layer_num = 24
        head_num = 16
    elif bert_size == "tiny_4":
        layer_num = 4
        head_num = 12
    elif bert_size == "tiny_6":
        layer_num = 6
        head_num = 12
    else: 
        layer_num = 12
        head_num = 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if bert_size == "large":
        model_dir = os.path.join(model_dir, "BERT_large")
        output_dir = os.path.join(output_dir, "BERT_large")

    # Processor & Task Info
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if task_name in default_params:
        batch_size = default_params[task_name]["batch_size"]
        max_seq_length = default_params[task_name]["max_seq_length"]
        eval_step = default_params[task_name]["eval_step"]
    
    model_dir = "models"
    output_dir = "output"

    teacher_model_dir = os.path.join(model_dir,task_name)
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(teacher_model_dir, do_lower_case=True)


    # Load Dataset
    data_dir = os.path.join("data",task_name)
    processed_data_dir = os.path.join(data_dir,'preprocessed')

    train_examples = processor.get_train_examples(data_dir)

    train_features = convert_examples_to_features(train_examples, label_list,
                                    max_seq_length, tokenizer, output_mode)

    train_features = train_features[:int(len(train_features) * args.sample_N)]
    print(f"Num examples = {len(train_features)}")

    train_data, train_labels = get_tensor_data(output_mode, train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    seed=42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mse_func = MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if bert_size == "large":
        model_dir = os.path.join(model_dir, "BERT_large")
        output_dir = os.path.join(output_dir, "BERT_large")

    # ================================================================================  #
    # Model Load
    # ================================================================================ #
    if args.kd_loss:
        teacher_model = BertForSequenceClassification.from_pretrained(teacher_model_dir, num_labels=num_labels)
        teacher_model.to(device)
        teacher_model.eval()
    else:
        teacher_model = None
    
    init_dir = os.path.join(model_dir,task_name) 
    model_dir = os.path.join(output_dir, task_name, "exploration", args.model_name)
    model_config = BertConfig.from_pretrained(model_dir)             

    quant_model_dir = os.path.join(output_dir, task_name, "exploration", args.quant_model_name)
    quant_config = BertConfig.from_pretrained(quant_model_dir)             

    if args.init:
        model = QuantBertForSequenceClassification.from_pretrained(init_dir, config = quant_config, num_labels=num_labels)
    else:
        model = QuantBertForSequenceClassification.from_pretrained(model_dir, config = model_config, num_labels=num_labels)
    
    model.to(device)

    print()
    print("==> Load Model DONE")
    print("==> Test Inference")

    if output_mode == "classification":
        loss_fct = CrossEntropyLoss()

    elif output_mode == "regression":
        loss_fct = MSELoss()

    from hessian import hessian
    from tqdm import tqdm

    tc_max_eigens = []
    for batch in tqdm(train_dataloader):
        input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
        hessian_comp = hessian(model, data=(input_ids, label_ids), criterion=loss_fct, cuda=True, input_zip = (input_ids, segment_ids, input_mask), teacher_model=teacher_model, kd_type=args.kd_loss_type)
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=3)
        tc_max_eigens = tc_max_eigens + top_eigenvalues

    pt_folder_name = "hessian_value_pts"
    file_name = args.file_name + ".pt" 

    file_dir = os.path.join(pt_folder_name, file_name)

    if not os.path.exists(pt_folder_name):
        os.mkdir(pt_folder_name)          

    print(file_dir)
    try:
        torch.save(tc_max_eigens, file_dir)
    except:
        import pdb; pdb.set_trace()
    print("==> Model Eigen Value DONE!")
    # st_max_eigens = []
    # for batch in tqdm(train_dataloader):
    #     input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
    #     hessian_comp = hessian(student_model, data=(input_ids, label_ids), criterion=loss_fct, cuda=True)
    #     top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=5)
    #     st_max_eigens = st_max_eigens + top_eigenvalues
    # print("==> Student Model Eigen Value DONE!")


if __name__ == "__main__":
    main()