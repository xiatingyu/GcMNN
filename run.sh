
export HF_ENDPOINT=https://hf-mirror.com

python sentence_bert_embeddings.py \
    --data_path 'data/WizardLM_evol_instruct_V2_143k.json' \
    --embedding_cache_path 'data/wizardlm_embeddings.npy' 

python GraphCut.py


model="../model/Qwen2-7B"
CUDA_VISIBLE_DEVICES=1 python FBNM-response.py \
        --start 0 --end 145000 \
        --base_model $model \
        --data_file "data/WizardLM_evol_instruct_V2_143k.json" \
        --output_file data/wizard/FBNM_qwen2_sentence.jsonl 


python select_data.py