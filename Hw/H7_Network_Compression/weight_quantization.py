import os
import torch
from Hw.H7_Network_Compression.utils import *


import argparse
import ast

parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str, help="the output checkpoints dir",
                    dest="checkpoint_dir")
parser.add_argument("--checkpoint_path", default="", type=str, help="the output checkpoints path",
                    dest="checkpoint_path")
args = parser.parse_args()

print("arguments:")
for arg in vars(args):
    print(arg, ":", getattr(args, arg))

print("-" * 100)

input_model_path = args.checkpoint_path
org_name = input_model_path.split('/')[-1].split('.')
output_model_path_encode16 = os.path.join(args.checkpoint_dir, '.'.join(org_name[:-1])+"_encode16."+org_name[-1])
output_model_path_encode8 = os.path.join(args.checkpoint_dir, '.'.join(org_name[:-1])+"_encode8."+org_name[-1])

print(f"\noriginal cost: {os.stat(input_model_path).st_size/1024/1024} MB.")
params = torch.load(input_model_path)
encode16(params, output_model_path_encode16)
print(f"16-bit cost: {os.stat(output_model_path_encode16).st_size/1024/1024} MB.")

params = torch.load(input_model_path)
encode8(params, output_model_path_encode8)
print(f"8-bit cost: {os.stat(output_model_path_encode8).st_size/1024/1024} MB.")
