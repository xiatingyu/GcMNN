import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from sentence_transformers import SentenceTransformer, util
import csv
from collections import Counter
import pickle
import time
import numpy as np
import pandas as pd
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--embedding_cache_path", type=str)
    args = parser.parse_args()
    print('------------------------------------------Embedding---------------------------------------------')
    

    model_name = 'paraphrase-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)

    with open(args.data_path, 'r') as f:
        data = json.load(f)

    corpus_sentences = []
    for item in data:
        tmp = 'Human:'
        tmp += item['conversations'][0]['value']
        tmp += '\nAssistant: '
        tmp += item['conversations'][1]['value']
        corpus_sentences.append(tmp)
    
    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)

    print("Store file on disc")
        
    np.save(f'{args.embedding_cache_path}', corpus_embeddings)