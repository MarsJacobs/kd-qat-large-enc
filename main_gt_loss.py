import os
import torch
import argparse
import copy

parser = argparse.ArgumentParser()

parser.add_argument("--task",
                        default='task',
                        type=str
                        )

parser.add_argument("--tau",
                        default=3.0,
                        type=float,
                        help="{0,1}")

args = parser.parse_args() 

model_1sb_dir = os.path.join("output", args.task, "quant", "1SB_save")
model_2sb_dir = os.path.join("output", args.task, "quant", "2SB_save")


model_1sb = torch.load(model_1sb_dir+ "/" + "pytorch_model.bin")
model_2sb = torch.load(model_2sb_dir+ "/" + "pytorch_model.bin")


model_1sb_ = copy.deepcopy(model_1sb)
model_2sb_ = copy.deepcopy(model_2sb)
model_ens_ = copy.deepcopy(model_1sb)



tau = torch.Tensor([args.tau])
tau = tau.to("cuda")
for n, p in model_1sb.items():
    result = p*tau + model_2sb[n]*(1.0-tau)
    model_ens_[n] = result

folder_dir = "mode_pths/" + args.task + "_" + str(args.tau) + "-1sb_" + str(1.0-args.tau) + "-2sb"

if not os.path.exists(folder_dir):
    os.mkdir(folder_dir)

torch.save(model_ens_, folder_dir + "/pytorch_model.bin")

print("==>SAVED  " + "mode_pths/" + args.task + "_" + str(args.tau) + "-1sb_" + str(1.0-args.tau) + "-2sb.bin")