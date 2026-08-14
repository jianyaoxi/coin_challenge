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
├── coinenv_requirements.txt
├── vllmenv_requirements.txt
└── results/               # Evaluation outputs from eval_model.py (.gzip.json)
# This will create virtual environments using your Python; ensure the created environment's Python version is >= 3.11
cd /root/coin_challenge
bash scripts/install.sh        # Creates coin_env (for evaluation scripts)
bash scripts/install_vllm.sh   # Creates vllm_env (for vLLM service)
DASHSCOPE_API_KEY="your Alibaba Cloud Bailian API Key"
nvidia-smi                       # 4× 2080Ti present
ls /root/coin_challenge/images   # Evaluation images exist
test -f ~/.env.ml && echo "key file ok"
cd /root/coin_challenge
bash scripts/launch_eval_qwen_oracle.sh   # Note: this assumes 4x 2080Ti GPUs are available; if not, adjust accordingly
cd ./coin_challenge
coin_env/bin/python eval_model.py 0 167 --description-type all
