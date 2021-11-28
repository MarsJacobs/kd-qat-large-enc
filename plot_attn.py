import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
import numpy
import os
import argparse
import random
import collections

def draw_map(fp_attn_weights, q_attn_weights, ft_attn_weights, word_list, type, seq_num, layer_num, head_num):
    folder = "Attention_weights/maps"
    
    map_dir = os.path.join(folder, f'seq_{seq_num}')
    if not os.path.exists(map_dir):
        os.mkdir(map_dir)
    map_dir = os.path.join(map_dir, f'layer_{layer_num}')
    if not os.path.exists(map_dir):
        os.mkdir(map_dir)
    # map_dir = os.path.join(map_dir, f'head_{head_num}')
    # if not os.path.exists(map_dir):
        # os.mkdir(map_dir)
    
    fp_attn_weights = fp_attn_weights.clone().detach().cpu().numpy()
    q_attn_weights = q_attn_weights.clone().detach().cpu().numpy()
    ft_attn_weights = ft_attn_weights.clone().detach().cpu().numpy()

    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(20,6))
    heatmap = ax1.pcolor(fp_attn_weights, cmap=plt.cm.Blues)
    
    ax1.set_xticks(numpy.arange(fp_attn_weights.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(numpy.arange(fp_attn_weights.shape[0]) + 0.5, minor=False)
    
    ax1.set_xlim(0, int(fp_attn_weights.shape[1]))
    ax1.set_ylim(0, int(fp_attn_weights.shape[0]))

    ax1.invert_yaxis()
    ax1.xaxis.tick_top()

    ax1.set_xticklabels(word_list, minor=False)
    ax1.set_yticklabels(word_list, minor=False)

    plt.xticks(rotation=45)

    heatmap = ax2.pcolor(q_attn_weights, cmap=plt.cm.Blues)

    ax2.set_xticks(numpy.arange(q_attn_weights.shape[1]) + 0.5, minor=False)
    ax2.set_yticks(numpy.arange(q_attn_weights.shape[0]) + 0.5, minor=False)

    ax2.set_xlim(0, int(q_attn_weights.shape[1]))
    ax2.set_ylim(0, int(q_attn_weights.shape[0]))

    ax2.invert_yaxis()
    ax2.xaxis.tick_top()

    ax2.set_xticklabels(word_list, minor=False)
    ax2.set_yticklabels(word_list, minor=False)

    plt.xticks(rotation=45)

    heatmap = ax3.pcolor(fp_attn_weights, cmap=plt.cm.Blues)

    ax3.set_xticks(numpy.arange(ft_attn_weights.shape[1]) + 0.5, minor=False)
    ax3.set_yticks(numpy.arange(ft_attn_weights.shape[0]) + 0.5, minor=False)

    ax3.set_xlim(0, int(ft_attn_weights.shape[1]))
    ax3.set_ylim(0, int(ft_attn_weights.shape[0]))

    ax3.invert_yaxis()
    ax3.xaxis.tick_top()

    ax3.set_xticklabels(word_list, minor=False)
    ax3.set_yticklabels(word_list, minor=False)

    plt.xticks(rotation=45)

    plt.savefig(map_dir +"/" + f"seq_{seq_num}_layer_{layer_num}_head_{head_num}.png")
    plt.close()

def main():
    # ================================================================================  #
    # Load
    # ================================================================================ #
    
    folder_name = "Attention_weights/pths"
    vocab = torch.load(folder_name + "/vocab_dict.pth")
    cola_input_batch = torch.load(folder_name + "/input_attn.pth")
    
    input_ids, input_mask, segment_ids, label_ids, seq_lengths = cola_input_batch

    fp_attn_dict = dict()
    q_attn_dict = dict()
    q_ft_attn_dict = dict()
    colored_string = ''
    

    for i in range(12):
        fp_attn_dict[f"layer_{i}"] = torch.load(folder_name+f"/FP_layer_{i}_attn_probs.pth")
        q_attn_dict[f"layer_{i}"] = torch.load(folder_name+f"/Q_layer_{i}_attn_probs.pth")
        q_ft_attn_dict[f"layer_{i}"] = torch.load(folder_name+f"/Q_FT_layer_{i}_attn_probs.pth")
    
    print("=========> LOAD COMPLETE!")

    for sentence_idx in range(16):
        word_list = list()    
        sentence_num = seq_lengths[sentence_idx].item()
        for word in range(sentence_num):
            word_list.append(vocab[input_ids[sentence_idx][word].item()])

        for layer_num in range(12):
            for head_num in range(12):
                print(f"=======> sentence {sentence_idx}, layer {layer_num}, head {head_num}")

                fp_attn_weights = fp_attn_dict[f'layer_{layer_num}'][sentence_idx][head_num][:sentence_num,:sentence_num]
                q_attn_weights = q_attn_dict[f'layer_{layer_num}'][sentence_idx][head_num][:sentence_num,:sentence_num] 
                q_ft_attn_weights = q_ft_attn_dict[f'layer_{layer_num}'][sentence_idx][head_num][:sentence_num,:sentence_num] 
                
                draw_map(fp_attn_weights, q_attn_weights, q_ft_attn_weights, word_list, "fp", sentence_idx, layer_num, head_num)
                #draw_map(q_attn_weights, word_list, "q", sentence_idx, layer_num, head_num)
                #draw_map(q_ft_attn_weights, word_list, "ft", sentence_idx, layer_num, head_num)
            
    print("=========> DONE!!")



if __name__ == "__main__":
    main()