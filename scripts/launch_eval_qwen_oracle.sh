#!/bin/bash
# ============================================================
# 启动 vLLM：AWQ 底座 + 微调 LoRA（Qwen2.5-VL-32B）
# 专为 RTX 2080 Ti (计算能力 7.5 / Turing) 优化
#
# 前置条件：
#   1. vllm_env 里 triton 必须是 3.2.0。
#      triton 3.3.1 在 Turing 上编译 LoRA 的 Triton 内核会失败
#      （vllm/lora/ops/triton_ops/lora_shrink_op.py），
#      运行过一次 `pip install triton==3.2.0` 即可。
#   2. 微调产物 LoRA adapter 在 /root/qwen_vl_finetune/output_v3
#      （QLoRA 训练 1086 步 / 2 epochs 后保存，原子单问数据集，
#        含 adapter_config.json + adapter_model.safetensors）
#
# 启动后：
#   - vLLM 服务端口: 3389
#   - 微调模型 model_id = "qwen32b"     <-- oracle / questioner 都用这个
#   - 未微调底座 model_id = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
# ============================================================

# --- 微调模型（LoRA 服务名）与 AWQ 底座 ---
# 注意：eval_model.py 的 oracle 已固定为远程 QwenOracle（千问内置模型，官方提交要求），
# 不再读取 ORACLE_MODEL_ID / --oracle / --local。下面的 export 仅保留做记录，不影响评测。
LORA_NAME="qwen32b"                                    # vLLM 里 LoRA 模块名
BASE_MODEL="Qwen/Qwen2.5-VL-32B-Instruct-AWQ"          # AWQ 底座（量化权重仅推理）
ADAPTER_DIR="jianyaoxi/qwen-lora-v7"         # 微调 LoRA adapter（1086 步最终产物）

export MY_VLLM_PORT=3389

# 提醒：若在另一个终端跑评测，需要先 export ORACLE_MODEL_ID="qwen32b"。
# Questioner.py 里 YourQuestioner 的默认 model_id 已是 "qwen32b"（微调 LoRA），
# 评测 questioner 会走微调模型；若想改用未微调的 AWQ 底座，手动传 model_id 覆盖即可。

# 激活虚拟环境
source /root/coin_challenge/vllm_env/bin/activate

echo "[INFO] triton: $(python -c 'import triton; print(triton.__version__)' 2>/dev/null) (Turing 需要 3.2.0)"

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

# 说明：
#   - max-model-len 3000：4x11GB 显存 + 20GB AWQ 权重后 KV cache 只够 ~3392 token，
#     3000 留安全余量；配合 utils.py 的 max_output_length=4000（max_tokens=1000）。
#   - mm-processor-kwargs：把 max_pixels 压到 512x28x28，避免 vLLM 为多模态预留
#     32768 token 的最坏边界（会超出 LoRA 元数据缓冲导致启动失败），
#     512x512 的训练图保持原生分辨率。

# 评测任务（在另一个终端运行；questioner 默认走本地 vLLM 的 qwen32b LoRA）
# 注意：
#   oracle 固定为远程 QwenOracle（千问内置模型，默认 qwen3.7-flash，可用 QWEN_MODEL_ID 覆盖），
#   国内直连，0.5~1.2s/次。--oracle / --local 参数已弃用，传入也不会改变行为。
#   ⚠ oracle 不能用微调的 qwen32b：它被训练成提问者，回答图像问题时只会输出提问者式拒绝。
#   ⚠ 不要用 qwen3.5-flash 当 oracle：推理模型，每次看图回答 20s+，评测会卡死。
# sleep 120
# source /root/coin_challenge/coin_env/bin/activate
# cd /root/coin_challenge

# # # 逐类跑：
# /root/coin_challenge/coin_env/bin/python eval_model.py 0 167 --description-type category
# /root/coin_challenge/coin_env/bin/python eval_model.py 0 167 --description-type color
# /root/coin_challenge/coin_env/bin/python eval_model.py 0 167 --description-type context
# /root/coin_challenge/coin_env/bin/python eval_model.py 0 167 --description-type color_feature
# /root/coin_challenge/coin_env/bin/python eval_model.py 0 167 --description-type color_context
# /root/coin_challenge/coin_env/bin/python eval_model.py 0 167 --description-type color_context_feature
