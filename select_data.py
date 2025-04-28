import json
import random
# 读取txt文件中的索引
indexes = []
with open('alpaca_test_bins_idx.txt', 'r', encoding='utf-8') as txt_file:
    for line in txt_file:
        indexes.append(list(line.strip().split()))

print(len(indexes))
print(indexes[0])
# 读取json文件
data = []
with open('FBNM_qwen_response.jsonl','r')as f:
    for line in f.readlines():
        data.append(json.loads(line))


extracted_data = []
original_indices = []
for index in indexes:
    tmp_with_index = [(data[int(j)], int(j)) for j in index]
    tmp = [item[0] for item in tmp_with_index]
    tmp = sorted(tmp, key=lambda x: x['fbnm'], reverse=True)
    sorted_tmp_with_index = sorted(tmp_with_index, key=lambda x: x[0]['fbnm'], reverse=True)
    extracted_data.extend([item[0] for item in sorted_tmp_with_index[:520]])
    original_indices.extend([item[1] for item in sorted_tmp_with_index[:520]])


print(len(extracted_data))

with open('submodular_alpaca_qwen2_7B.json','w') as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=4)

with open('submodular_alpaca_qwen2_7B.txt','w') as f:
    for id in original_indices:
        f.write(str(id)+'\n')
#random select
# extracted_data = []
# for index in indexes:
#     tmp_with_index = [(data[int(j)], int(j)) for j in index]
#     tmp = [item[0] for item in tmp_with_index]
#     random.shuffle(tmp)
#     extracted_data.extend(tmp[:1000])
# print(len(extracted_data))
# with open('submodular_alpaca_qwen_gc.json','w') as f:
#     json.dump(extracted_data, f, ensure_ascii=False, indent=4)