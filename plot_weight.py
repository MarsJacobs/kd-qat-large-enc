import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
import numpy
import os
import argparse
import random
import collections

folder_dir = 'models/FFN'
plt_dir = 'plt_storage/weight_comp'

bert_base = torch.load(folder_dir + "/" + "bert_base.bin", map_location='cpu')
cola_base = torch.load(folder_dir + "/" + "cola_dw.bin", map_location='cpu')
cola_ffn = torch.load(folder_dir + "/" + "FFN_KD.bin", map_location='cpu')

# Query
query_dir = os.path.join(plt_dir, "query")
key_dir = os.path.join(plt_dir, "key")
value_dir = os.path.join(plt_dir, "value")
ffn1_dir = os.path.join(plt_dir, "ffn1")
ffn2_dir = os.path.join(plt_dir, "ffn2")

for layer in range(12):
    print(f"====> layer_{layer} Start!")
    fig, [ax1,ax2, ax3] = plt.subplots(1,3, figsize=(24,6))
    name = "query"
    ax1.hist(bert_base[f"bert.encoder.layer.{layer}.attention.self.{name}.weight"].reshape(-1).detach().cpu().numpy(), bins=100, label='bert_base', color="dodgerblue")
    ax1.set_title(f"{layer}th layer BERT base {name} Weight")

    ax2.hist(cola_base[f"bert.encoder.layer.{layer}.attention.self.{name}.weight"].reshape(-1).detach().cpu().numpy(), bins=100, label='cola_base', color="dodgerblue")
    ax2.set_title(f"{layer}th layer COLA base {name} Weight")

    ax3.hist(cola_ffn[f"bert.encoder.layer.{layer}.attention.self.{name}.weight"].reshape(-1).detach().cpu().numpy(), bins=100, label='cola_ffn', color="dodgerblue")
    ax3.set_title(f"{layer}th layer COLA FFN {name} Weight")
    plt.savefig(query_dir + f"/layer_{layer}")
    plt.close()

    fig, [ax1,ax2, ax3] = plt.subplots(1,3, figsize=(24,6))
    name = "key"
    ax1.hist(bert_base[f"bert.encoder.layer.{layer}.attention.self.{name}.weight"].reshape(-1).detach().cpu().numpy(), bins=100, label='bert_base', color="dodgerblue")
    ax1.set_title(f"{layer}th layer BERT base {name} Weight")

    ax2.hist(cola_base[f"bert.encoder.layer.{layer}.attention.self.{name}.weight"].reshape(-1).detach().cpu().numpy(), bins=100, label='cola_base', color="dodgerblue")
    ax2.set_title(f"{layer}th layer COLA base {name} Weight")

    ax3.hist(cola_ffn[f"bert.encoder.layer.{layer}.attention.self.{name}.weight"].reshape(-1).detach().cpu().numpy(), bins=100, label='cola_ffn', color="dodgerblue")
    ax3.set_title(f"{layer}th layer COLA FFN {name} Weight")
    plt.savefig(key_dir + f"/layer_{layer}")
    plt.close()

    fig, [ax1,ax2, ax3] = plt.subplots(1,3, figsize=(24,6))
    name = "value"
    ax1.hist(bert_base[f"bert.encoder.layer.{layer}.attention.self.{name}.weight"].reshape(-1).detach().cpu().numpy(), bins=100, label='bert_base', color="dodgerblue")
    ax1.set_title(f"{layer}th layer BERT base {name} Weight")

    ax2.hist(cola_base[f"bert.encoder.layer.{layer}.attention.self.{name}.weight"].reshape(-1).detach().cpu().numpy(), bins=100, label='cola_base', color="dodgerblue")
    ax2.set_title(f"{layer}th layer COLA base {name} Weight")

    ax3.hist(cola_ffn[f"bert.encoder.layer.{layer}.attention.self.{name}.weight"].reshape(-1).detach().cpu().numpy(), bins=100, label='cola_ffn', color="dodgerblue")
    ax3.set_title(f"{layer}th layer COLA FFN {name} Weight")
    plt.savefig(value_dir + f"/layer_{layer}")
    plt.close()

print("===> DONE!")