from apricot import FeatureBasedSelection, FacilityLocationSelection, GraphCutSelection
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse
import json

def external_retrival(embeddings, sample_number):
    selector = GraphCutSelection(sample_number, metric='cosine', optimizer='naive')
    print(embeddings.shape)
    selector.fit(embeddings)
    data_index = selector.ranking[:sample_number]
    return data_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--data_path", type=int, default='data/WizardLM_evol_instruct_V2_143k.json')
    parser.add_argument("--embed_path", type=int, default='data/wizard_embeddings.npy')
    parser.add_argument("--output_path", type=int, default='wizard_test_bins_idx.txt')
    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        data = json.load(f)
    total_len = len(data)

    k = args.k
    n = int(len(data))
    budget_n = len(data) // k

    print(f"total: {len(data)} n: {n}, k: {k}, budget_n: {budget_n}")

    embeddings_original = np.load(args.embed_path)
    embeddings = embeddings_original.copy()
    indices_original = np.arange(len(data))
    remaining_indices = indices_original.copy()

    with open(args.output_path, 'w') as f:
        bins = []
        for i in range(k):
            print(f"bin {i}/{k}")
            # 调用 external_retrival 函数
            result_indices = external_retrival(embeddings, budget_n)

            # 将缩减后数据的索引转换为原始数据的索引
            result_indices_original = remaining_indices[result_indices]
            result_indices_original = np.sort(result_indices_original)
            bins.append(result_indices_original)
            f.write('\t'.join(map(str, result_indices_original)) + '\n')

            # 更新 remaining_indices 和 embeddings
            remaining_indices = np.delete(indices_original, np.concatenate(bins))
            embeddings = embeddings_original[remaining_indices]