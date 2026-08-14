# CoIN Challenge Evaluation Project

This repository hosts the evaluation code and runtime environment for the CoIN Challenge, including the evaluation pipeline, vLLM inference service, dataset, and deployment scripts.

# Directory Structure
```text
coin_challenge
├── Questioner.py          # Questioner interface + fine-tuned YourQuestioner (questioner)
├── Oracle.py              # OracleInterface abstract base class (required by official submission)
├── oracle_qwen.py         # QwenOracle: Oracle implementation using remote Qwen vision model (currently fixed)
├── env.py                 # QAEnv: gym environment, manages episode/candidate images/scoring, calls Oracle
├── utils.py               # Clients (QwenLLM / ClientBasedLLM / GeminiLLM) + utilities
├── eval_model.py          # Main evaluation entry point
├── QA_eval/
│   ├── episodes_train.jsonl   # 168 episodes (paths/task descriptions/candidate images + match labels)
│   └── results/               # (provided) reference evaluation outputs
├── images/                # Target images / candidate images
├── scripts/
│   ├── install.sh             # Create coin_env (evaluation dependencies)
│   ├── install_vllm.sh        # Create vllm_env (vLLM service dependencies)
│   └── launch_eval_qwen_oracle.sh  # Start vLLM: AWQ base + fine-tuned LoRA (questioner)
├── coin_env/              # Python virtual environment for evaluation
├── vllm_env/              # Python virtual environment for vLLM service
└── results/               # Evaluation outputs from eval_model.py (.gzip.json)
```
# Environment Installation
The scripts will create virtual environments based on your system Python. 


Creates coin_env for evaluation scripts

> source scripts/install.sh        

Creates vllm_env for vLLM service

> source scripts/install_vllm.sh   

# Start vLLM Inference Service

This script is configured with `CUDA_VISIBLE_DEVICES=0,1,2,3` and `--tensor-parallel-size 4`, which is the setup for my local machine. Please adjust these parameters according to the number of GPUs available on your device.

> source scripts/launch_eval_qwen_oracle.sh

# Run Evaluation

The test images should go in the folder image  and episodes_test.jsonl should go in the folder QA_eval （test_result will be in this folder）
> coin_env/bin/python eval_model.py 0 167 --description-type all
