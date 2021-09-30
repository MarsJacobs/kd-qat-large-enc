import torch
import copy
model_fr = torch.load("SST_FP.pt")
model_fr = model_fr['model']
model_to = torch.load("pytorch_model.bin", map_location='cpu')

model_fr_dict = copy.deepcopy(model_fr)
model_to_dict = copy.deepcopy(model_to)

new_dict = dict()

for name, param in model_fr_dict.items():
    to_list = []
    to_list.append("bert")
    from_list = name.split('.')

    if from_list[0] == "encoder":
        if 'emb' in from_list[2]:
            to_list.append("embeddings")
            if 'tokens' in from_list[2]:
                to_list.append("word_embeddings")
            elif "positions" in from_list[2]:
                to_list.append("position_embeddings")
            elif "emb_layer_norm" in from_list[2]:
                to_list.append("LayerNorm")
            
        
        elif 'layers' in from_list[2]:
            to_list.append("encoder")
            to_list.append("layer")
            to_list.append(from_list[3])

            if 'self_attn' == from_list[4]:
                to_list.append("attention")
                if 'q_proj' == from_list[5]:
                    to_list.append("self")
                    to_list.append("query")
                if 'k_proj' == from_list[5]:
                    to_list.append("self")
                    to_list.append("key")
                if 'v_proj' == from_list[5]:
                    to_list.append("self")
                    to_list.append("value")
                if 'out_proj' == from_list[5]:
                    to_list.append("output")
                    to_list.append("dense")                    
            
            if 'self_attn_layer_norm' == from_list[4]:
                to_list.append("attention")
                to_list.append("output")
                to_list.append("LayerNorm")

            if 'fc1' in from_list[4]:
                to_list.append("intermediate")
                to_list.append("dense")
            
            if 'fc2' in from_list[4]:
                to_list.append("output")
                to_list.append("dense")

            if 'final_layer_norm' in from_list[4]:
                to_list.append("output")
                to_list.append("LayerNorm")

    elif from_list[0] == "classification_heads":
        
        if from_list[2] == "dense":
            to_list.append("pooler")
            to_list.append("dense")
        
        elif from_list[2] == "out_proj":
            to_list[0] = "classifier"

    if 'weight' in from_list[-1]:
        to_list.append("weight")
    elif 'bias' in from_list[-1]:
        to_list.append("bias")
    
    name = ""
    for word in to_list:
        name += word
        name += "."
    name = name[:-1]
    
    new_dict[name] = param

for name, param in model_to_dict.items():
    if name in new_dict:
        print("success! {} -> {}".format(param.shape, new_dict[name].shape))
    else:
        print("Fail {} is not exist in dict".format(name))
import pdb; pdb.set_trace()

    


        
        
    


