# GcMNN

## Environment Setup

Before starting, ensure your environment is properly configured:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Hugging Face Mirror (Optional):**
   If you are in mainland China, it is recommended to use the Hugging Face mirror site to accelerate downloads:
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. **Clone the llama-factory Repository:**
   The model training process requires the [llama-factory](https://github.com/hiyouga/LLaMA-Factory) script. Clone the repository and follow its documentation to complete the installation:
   ```bash
   git clone https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory
   pip install -e .
   ```
4. **Download the datasets:**
  The datasets can be download here, [Evol-Instruct](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k) and [Alpaca-GPT4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4)

## Data Processing Workflow

### 1. Generate Sentence-BERT Embeddings
Use the `sentence_bert_embeddings.py` script to generate embeddings for the input data:
```bash
python sentence_bert_embeddings.py \
    --data_path 'data/WizardLM_evol_instruct_V2_143k.json' \
    --embedding_cache_path 'data/wizardlm_embeddings.npy'
```
This step reads the input data file and generates an embedding cache file.

### 2. Graph Cut Processing
Run the `GraphCut.py` script to perform graph cut operations on the embeddings:
```bash
python GraphCut.py
```
This step performs clustering or segmentation on the embedding data.

### 3. Generate Model Responses
Use the `FBNM-response.py` script to calculate the MNN values of all the data. Mainly, Step 2 and Step 3 can be processed in parallel. 
```bash
model="../model/Qwen2-7B"
CUDA_VISIBLE_DEVICES=1 python FBNM-response.py \
        --start 0 --end 145000 \
        --base_model $model \
        --data_file "data/WizardLM_evol_instruct_V2_143k.json" \
        --output_file data/wizard/FBNM_qwen2_sentence.jsonl
```
**Notes:**
- The `--start` and `--end` parameters specify the range of data to process.
- The `--base_model` parameter specifies the path to the base model.
- The `--data_file` parameter is the input data file.
- The `--output_file` parameter is the generated output file.

### 4. Data Selection
Finally, run the `select_data.py` script to filter the data:
```bash
python select_data.py
```

## Key Tools and Resources

- **llama-factory:** The core tool for model training. Refer to its official documentation for more details:
  - GitHub Repository: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
  - Installation Guide: [Installation Guide](https://github.com/hiyouga/LLaMA-Factory#installation)
  - The version we used during the experiment is version 0.8.2. Please refer to the `train.sh` script for the model training. 


## Citation
If you finding our work interesting or helpful to you, please cite this repo.
```
@article{xia2025selective,
  title={Selective fine-tuning for large language models via matrix nuclear norm},
  author={Xia, Tingyu and Li, Yahan and Wu, Yuan and Chang, Yi},
  journal={Information Processing \& Management},
  volume={62},
  number={6},
  pages={104259},
  year={2025},
  publisher={Elsevier}
}
```


