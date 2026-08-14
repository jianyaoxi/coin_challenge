
LORA_NAME="qwen32b"                                  
BASE_MODEL="Qwen/Qwen2.5-VL-32B-Instruct-AWQ"         
ADAPTER_DIR="jianyaoxi/qwen-lora-v7"         
export MY_VLLM_PORT=3389

source ./vllm_env/bin/activate

# 启动服务
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve ${BASE_MODEL} \
    --port ${MY_VLLM_PORT} \
    --tensor-parallel-size 4 \
    --disable-custom-all-reduce \
    --dtype float16 \
    --max-model-len 10240 \
    --mm-processor-kwargs '{"max_pixels": 401408, "min_pixels": 3136}' \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --trust-remote-code \
    --enable-lora \
    --lora-modules "${LORA_NAME}=${ADAPTER_DIR}"

# /root/coin_challenge/coin_env/bin/python eval_model.py 0 167 --description-type category
# /root/coin_challenge/coin_env/bin/python eval_model.py 0 167 --description-type color
# /root/coin_challenge/coin_env/bin/python eval_model.py 0 167 --description-type context
# /root/coin_challenge/coin_env/bin/python eval_model.py 0 167 --description-type color_feature
# /root/coin_challenge/coin_env/bin/python eval_model.py 0 167 --description-type color_context
# /root/coin_challenge/coin_env/bin/python eval_model.py 0 167 --description-type color_context_feature
