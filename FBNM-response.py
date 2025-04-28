import csv
import os
import time
import numpy as np
from regex import F
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Model, AutoModel
import torch
import math
import json
from tqdm import tqdm
import random
import argparse
import copy
import torch.nn.functional as F
import torch.nn as nn


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def normalize(R):
    with torch.no_grad():
        mean = R.mean(dim=0)
        R = R - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R / norms
    return R

def fast_nuclear_norm(X, D=None):
    if D is None:
        D = min(X.shape[0], X.shape[1])

    # 计算每列的平方和，然后开方得到L2范数
    l2_norms = torch.sqrt(torch.sum(torch.pow(X, 2), dim=0))

    # 降序排序L2范数
    list_svd, _ = torch.sort(l2_norms, descending=True)

    # 取排序后的前D个范数
    top_D_l2_norms = list_svd[:D]

    # 计算FBNM近似值
    L_FBNM = torch.sum(top_D_l2_norms)
    FBNM_normalize = L_FBNM.item()

    return FBNM_normalize



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default='model/Qwen1.5-1.8B')  # model path
    parser.add_argument("--data_file", type=str, default='/cpfs01/shared/Group-m6/dangkai.dk/workspace/scripts/code_evol_pack_final.jsonl')  # data path
    parser.add_argument("--output_file", type=str, default='tmp.jsonl')  # output path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=10)  # end index
    parser.add_argument("--type", type=str, default="base")  # end index
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("Processing data ...")
    instructions, systems, outputs = [], [], []

    data = []
    with open(args.data_file,"r", encoding="utf8") as f:
        data = json.load(f)

    already_processed = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as reader:
            for item in reader.readlines():
                already_processed.append(json.loads(item, strict=False))
        
    if already_processed != []:
        args.start = already_processed[-1]['id'] + 1
    
    
    if args.start >= args.end:
        print("all data is processed!")
        sys.exit(0)
    else:
        print(f"start from {args.start} to {args.end}")
        
    data = data[args.start:args.end]


    PROMPT_RAW = {
        "user": (
            "Human: {query}\nAssistant: "    
        ),
        "ass": (
            "{response}"
        )
    }

    PROMPT_CHATML = {
        "system": (
            "<|im_start|>system\n{system}<|im_end|>"
        ),
        "user": (
            "\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n" 
        ),
        "ass": (
            "{response}<|im_end|>"
        )
    }

    data_list = []
    for item in data:
        encoded_messages = ""
        for i in range(len(item['conversations'])-1):
            message = item['conversations'][i]
            if args.type == "chatml":
                if message["from"] == "human":
                    encoded_messages += PROMPT_CHATML['user'].format(query=message['value'])
                else:
                    encoded_messages += PROMPT_CHATML['ass'].format(response=message['value'])
            else:
                if message["from"] == "human":
                    encoded_messages += PROMPT_RAW['user'].format(query=message['value'])
                else:
                    encoded_messages += PROMPT_RAW['ass'].format(response=message['value'])
        encoded_messages = encoded_messages.strip()
        response = PROMPT_RAW['ass'].format(response=item['conversations'][-1]['value'])
        data_list.append((encoded_messages,response))

    print(len(data_list))
    print("Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModel.from_pretrained(args.base_model, device_map="auto", output_hidden_states=True).eval()

   

    fbnm = []
    with open(args.output_file, "w+", encoding="utf8") as f:
        for i in tqdm(range(len(data_list))):
            messages = data_list[i][0]
            input_ids = tokenizer(messages, truncation=True, max_length=4096, return_tensors="pt")
            index = torch.count_nonzero(torch.squeeze(input_ids['input_ids'])).item()

            messages = data_list[i][1]
            token_ids = tokenizer(messages, truncation=True, max_length=4096-index, return_tensors="pt")
            input_ids = torch.cat((input_ids['input_ids'].long(), token_ids['input_ids'].long()), dim=-1).to(device)

            with torch.no_grad():
                # print(model.device, input_ids.device)
                output = model(input_ids)[0]
                output = output[0, index:, :]
                
                r = normalize(output)
                # 计算FBNM
                FBNM = fast_nuclear_norm(r)
                fbnm.append(FBNM)
                

            tmp = copy.deepcopy(data[i])
            tmp['id'] = i+int(args.start)
            tmp['fbnm'] = FBNM
            json.dump(tmp, f, ensure_ascii=False)
            f.write("\n")
        
            