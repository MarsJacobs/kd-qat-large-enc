import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import argparse
import random
import collections

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                            default='cola',
                            type=str,
                            help="The name of the task to train.")
    args = parser.parse_args() 

    batch_size = 16 if args.task_name=="cola" else 32
    fig_size = 72 if args.task_name =="rte" else 24

    folder_name = args.task_name.upper() + '_KD_data'
    
    vocab_file = "vocab.txt"
    vocab = load_vocab(vocab_file)

    plt_folder_name = os.path.join(folder_name, 'plt_graphs')
    
    if not os.path.exists(plt_folder_name):
        os.mkdir(plt_folder_name)

    FP_input_dict = dict()
    FP_output_dict = dict()
    FP_Loutput_dict = dict()

    Q_input_dict = dict()
    Q_output_dict = dict()
    Q_Loutput_dict = dict()
    
    input_sentence = torch.load(folder_name+ "/" +"pths/"+ args.task_name + "_input_batch.pt")
    
    # Load pt Data File
    for i in range(12):
        # FP_input_dict[f"input_{i}"] = torch.load(folder_name+f"/pths/FP_layer_{i}_ffn1_input.pt")
        # FP_output_dict[f"output_{i}"] = torch.load(folder_name+f"/pths/FP_layer_{i}_ffn2_output.pt")
        # FP_Loutput_dict[f"Loutput_{i}"] = torch.load(folder_name+f"/pths/FP_layer_{i}_ffn2_Layernorm_output.pt")

        Q_input_dict[f"input_{i}"] = torch.load(folder_name+f"/pths/Q_layer_{i}_ffn1_input.pt")
        Q_output_dict[f"output_{i}"] = torch.load(folder_name+f"/pths/Q_layer_{i}_ffn2_output.pt")
        Q_Loutput_dict[f"Loutput_{i}"] = torch.load(folder_name+f"/pths/Q_layer_{i}_ffn2_Layernorm_output.pt")
    
    sentence_idx = 13#random.choice(range(batch_size))
    print("===>LOAD COMPLETE!")
    
    for i in range(12):
        
        seq_length = input_sentence[-1][sentence_idx]
        word_idx = random.choice(range(seq_length))
        histogram_folder_name = os.path.join(plt_folder_name, "histogram")
        if not os.path.exists(histogram_folder_name):
            os.mkdir(histogram_folder_name)
        
        for word in range(seq_length):
            word_name = vocab[input_sentence[0][sentence_idx][word].item()]
            # input, output, Loutput Activation Histogram    
            # fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=(24,6))
            # ax1.hist(FP_input_dict[f"input_{i}"][sentence_idx][word].reshape(-1).detach().cpu().numpy(), bins=100, label="input")
            # ax1.set_title(f"{i}th Layer Word input : " + word_name)
            
            # ax2.hist(FP_output_dict[f"output_{i}"][sentence_idx][word].reshape(-1).detach().cpu().numpy(), bins=100, label="output")
            # ax2.set_title(f"{i}th Layer Word output : " + word_name)
            
            # ax3.hist(FP_Loutput_dict[f"Loutput_{i}"][sentence_idx][word].reshape(-1).detach().cpu().numpy(), bins=100, label="Loutput")
            # ax3.set_title(f"{i}th Layer Word Loutput : " + word_name)

            # plt.savefig(histogram_folder_name + f"/FP_{i}th_{word}th_word_layer")
            # plt.close(fig)

            fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=(24,6))
            ax1.hist(Q_input_dict[f"input_{i}"][sentence_idx][word].reshape(-1).detach().cpu().numpy(), bins=100, label="input")
            ax1.set_title(f"{i}th Layer Word input : " + word_name)
            
            ax2.hist(Q_output_dict[f"output_{i}"][sentence_idx][word].reshape(-1).detach().cpu().numpy(), bins=100, label="output")
            ax2.set_title(f"{i}th Layer Word output : " + word_name)
            
            ax3.hist(Q_Loutput_dict[f"Loutput_{i}"][sentence_idx][word].reshape(-1).detach().cpu().numpy(), bins=100, label="Loutput")
            ax3.set_title(f"{i}th Layer Word Loutput : " + word_name)
            
            plt.savefig(histogram_folder_name + f"/Q_{i}th_{word}th_word_layer")
            plt.close(fig)

        
        # FP_input = FP_input_dict[f"input_{i}"][sentence_idx] # Dim : (max_seq_length, 768)
        # FP_output = FP_output_dict[f"output_{i}"][sentence_idx]
        # FP_Loutput = FP_Loutput_dict[f"Loutput_{i}"][sentence_idx]
        
        x_axis = []
        # F_in_min_axis = []; F_in_max_axis = []; F_in_mean_axis = []; F_in_p_std = []; F_in_n_std = []
        # F_o_min_axis = []; F_o_max_axis = []; F_o_mean_axis = []; F_o_p_std = []; F_o_n_std = []
        # F_l_min_axis = []; F_l_max_axis = []; F_l_mean_axis = []; F_l_p_std = []; F_l_n_std = []

        # fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=(fig_size,6))
        for j in range(input_sentence[-1][sentence_idx]):
            word = vocab[input_sentence[0][sentence_idx][j].item()]
            if word in x_axis :
                word = word + "_"
            x_axis.append(word)
        #     F_in_min_axis.append(FP_input[j].min())
        #     F_in_max_axis.append(FP_input[j].max())
        #     F_in_mean_axis.append(FP_input[j].mean())
        #     F_in_p_std.append(FP_input[:input_sentence[-1][sentence_idx]].std().item()*6)
        #     F_in_n_std.append(FP_input[:input_sentence[-1][sentence_idx]].std().item()*-6)

        #     F_o_min_axis.append(FP_output[j].min())
        #     F_o_max_axis.append(FP_output[j].max())
        #     F_o_mean_axis.append(FP_output[j].mean())
        #     F_o_p_std.append(FP_output[:input_sentence[-1][sentence_idx]].std().item()*6)
        #     F_o_n_std.append(FP_output[:input_sentence[-1][sentence_idx]].std().item()*-6)

        #     F_l_min_axis.append(FP_Loutput[j].min())
        #     F_l_max_axis.append(FP_Loutput[j].max())
        #     F_l_mean_axis.append(FP_Loutput[j].mean())
        #     F_l_p_std.append(FP_Loutput[:input_sentence[-1][sentence_idx]].std().item()*6)
        #     F_l_n_std.append(FP_Loutput[:input_sentence[-1][sentence_idx]].std().item()*-6)
        
        # #print("===> Sentence ")
        # #print(x_axis)
    
        # ax1.plot(x_axis, F_in_min_axis, 'o-', color = 'dodgerblue', label = 'min')
        # ax1.plot(x_axis, F_in_max_axis, 'o-', color = 'r', label = 'max')
        # ax1.plot(x_axis, F_in_mean_axis, color = 'gray', label = 'mean')
        # ax1.plot(x_axis, F_in_p_std,  color = 'lightgray', linestyle='dashed', label = '6*std')
        # ax1.plot(x_axis, F_in_n_std,  color = 'lightgray', linestyle='dashed')
        # fig.autofmt_xdate()
        # ax1.legend(loc='lower right')
        # ax1.set_title(f"{i}th layer FP input min-max Graph")

        # ax2.plot(x_axis, F_o_min_axis, 'o-', color = 'dodgerblue', label = 'min')
        # ax2.plot(x_axis, F_o_max_axis, 'o-', color = 'r', label = 'max')
        # ax2.plot(x_axis, F_o_mean_axis, color = 'gray', label = 'mean')
        # ax2.plot(x_axis, F_o_p_std,  color = 'lightgray', linestyle='dashed', label = '6*std')
        # ax2.plot(x_axis, F_o_n_std,  color = 'lightgray', linestyle='dashed')
        # fig.autofmt_xdate()
        # ax2.legend(loc='lower right')
        # ax2.set_title(f"{i}th layer FP output min-max Graph")

        # ax3.plot(x_axis, F_l_min_axis, 'o-', color = 'dodgerblue', label = 'min')
        # ax3.plot(x_axis, F_l_max_axis, 'o-', color = 'r', label = 'max')
        # ax3.plot(x_axis, F_l_mean_axis, color = 'gray', label = 'mean')
        # ax3.plot(x_axis, F_l_p_std,  color = 'lightgray', linestyle='dashed', label = '6*std')
        # ax3.plot(x_axis, F_l_n_std,  color = 'lightgray', linestyle='dashed')
        # fig.autofmt_xdate()
        # ax3.legend(loc='lower right')
        # ax3.set_title(f"{i}th layer FP Loutput min-max Graph")
        
        # plt.savefig(plt_folder_name + f"/FP_mm_{i}th_layer")
        # plt.close(fig)
        
        
        Q_input = Q_input_dict[f"input_{i}"][sentence_idx] # Dim : (max_seq_length, 768)
        Q_output = Q_output_dict[f"output_{i}"][sentence_idx]
        Q_Loutput = Q_Loutput_dict[f"Loutput_{i}"][sentence_idx]
        
        
        Q_in_min_axis = []; Q_in_max_axis = []; Q_in_mean_axis = []; Q_in_p_std = []; Q_in_n_std = []
        Q_o_min_axis = []; Q_o_max_axis = []; Q_o_mean_axis = []; Q_o_p_std = []; Q_o_n_std = []
        Q_l_min_axis = []; Q_l_max_axis = []; Q_l_mean_axis = []; Q_l_p_std = []; Q_l_n_std = []

        fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=(fig_size,6))
        for j in range(input_sentence[-1][sentence_idx]):
    
            Q_in_min_axis.append(Q_input[j].min())
            Q_in_max_axis.append(Q_input[j].max())
            Q_in_mean_axis.append(Q_input[j].mean())
            Q_in_p_std.append(Q_input[:input_sentence[-1][sentence_idx]].std().item()*6)
            Q_in_n_std.append(Q_input[:input_sentence[-1][sentence_idx]].std().item()*-6)

            Q_o_min_axis.append(Q_output[j].min())
            Q_o_max_axis.append(Q_output[j].max())
            Q_o_mean_axis.append(Q_output[j].mean())
            Q_o_p_std.append(Q_output[:input_sentence[-1][sentence_idx]].std().item()*6)
            Q_o_n_std.append(Q_output[:input_sentence[-1][sentence_idx]].std().item()*-6)

            Q_l_min_axis.append(Q_Loutput[j].min())
            Q_l_max_axis.append(Q_Loutput[j].max())
            Q_l_mean_axis.append(Q_Loutput[j].mean())
            Q_l_p_std.append(Q_Loutput[:input_sentence[-1][sentence_idx]].std().item()*6)
            Q_l_n_std.append(Q_Loutput[:input_sentence[-1][sentence_idx]].std().item()*-6)
        
        ax1.plot(x_axis, Q_in_min_axis, 'o-', color = 'dodgerblue', label = 'min')
        ax1.plot(x_axis, Q_in_max_axis, 'o-', color = 'r', label = 'max')
        ax1.plot(x_axis, Q_in_mean_axis, color = 'gray', label = 'mean')
        ax1.plot(x_axis, Q_in_p_std,  color = 'lightgray', linestyle='dashed', label = '6*std')
        ax1.plot(x_axis, Q_in_n_std,  color = 'lightgray', linestyle='dashed')
        fig.autofmt_xdate()
        ax1.legend(loc='lower right')
        ax1.set_title(f"{i}th layer Q input min-max Graph")

        ax2.plot(x_axis, Q_o_min_axis, 'o-', color = 'dodgerblue', label = 'min')
        ax2.plot(x_axis, Q_o_max_axis, 'o-', color = 'r', label = 'max')
        ax2.plot(x_axis, Q_o_mean_axis, color = 'gray', label = 'mean')
        ax2.plot(x_axis, Q_o_p_std,  color = 'lightgray', linestyle='dashed', label = '6*std')
        ax2.plot(x_axis, Q_o_n_std,  color = 'lightgray', linestyle='dashed')
        fig.autofmt_xdate()
        ax2.legend(loc='lower left')
        ax2.set_title(f"{i}th layer Q output min-max Graph")

        ax3.plot(x_axis, Q_l_min_axis, 'o-', color = 'dodgerblue', label = 'min')
        ax3.plot(x_axis, Q_l_max_axis, 'o-', color = 'r', label = 'max')
        ax3.plot(x_axis, Q_l_mean_axis, color = 'gray', label = 'mean')
        ax3.plot(x_axis, Q_l_p_std,  color = 'lightgray', linestyle='dashed', label = '6*std')
        ax3.plot(x_axis, Q_l_n_std,  color = 'lightgray', linestyle='dashed')
        fig.autofmt_xdate()
        ax3.legend(loc='lower right')
        ax3.set_title(f"{i}th layer Q Loutput min-max Graph")
        
        plt.savefig(plt_folder_name + f"/Q_mm_{i}th_layer") 
        plt.close(fig)
        print(f"===>LAYER_{i} COMPLETE!")
        


if __name__ == "__main__":
    main()