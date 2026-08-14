#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QwenOracle —— 基于阿里云百炼（DashScope）千问视觉模型的 Oracle 实现。

继承自 OracleInterface（见 Oracle.py），实现 `ask(prompt, images)`：
把 Questioner 的问题 + 目标图像编码为 OpenAI 兼容的多模态消息，调用千问
视觉模型，流式拼出回答并返回。Oracle 永远对目标图如实回答 —— 它是
Questioner 唯一可信的信息来源。

用法（可在自己的评测 harness 里替换 oracle，或在 eval_model.py 的
`oracle_client = ...` 处换成本类）：

    from oracle_qwen import QwenOracle
    oracle = QwenOracle()                      # 默认 qwen3.7-flash，国内直连、快
    # oracle = QwenOracle(model_id="qwen-vl-max")   # 更强模型
    answer = oracle.ask("What is the color of the object in the target image?",
                        images=[pil_image_or_ndarray])

依赖：~/.env.ml 里的 DASHSCOPE_API_KEY（load_api_keys() 已加载；密钥绝不打印）。
可用环境变量 QWEN_MODEL_ID 覆盖默认模型。

与 env.py 的配合：env.py 的 `ask_oracle` 已带 @retry(stop_max_attempt_number=5,
wait_fixed=80000)，所以这里**不**做内部重试，失败直接抛错由外层兜底，
避免重试时间叠加。
"""
import base64
import os
from io import BytesIO
from typing import Iterable, Union

import numpy as np
import openai
from PIL import Image

from Oracle import OracleInterface
from utils import load_api_keys

# DashScope OpenAI 兼容 endpoint（走国内网络，与 utils.QwenLLM 一致）
_DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 图像边长上限（与 utils.py 的防护一致）：超出则等比缩小，保证 oracle 总能出图
_SAFEGUARD_IMAGE_RESOLUTION = 1024
# 提示词长度上限（4 字母≈1 token，约 2 万 token）
_SAFEGUARD_N_LETTERS = 80_000


def _encode_image_b64(image: Image.Image, fmt: str) -> str:
    buf = BytesIO()
    image.save(buf, format=fmt.upper())
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class QwenOracle(OracleInterface):
    """Oracle：用千问（DashScope）视觉模型对目标图如实回答。"""

    def __init__(
        self,
        model_id: str = None,
        temperature: float = 1e-6,
        max_output_length: int = 4000,
    ):
        # 从 ~/.env.ml 加载密钥（DASHSCOPE_API_KEY）
        load_api_keys()
        self.model_id = model_id or os.environ.get("QWEN_MODEL_ID", "qwen3.7-flash")
        self.temperature = temperature
        self.max_output_length = max_output_length

        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY 缺失：请在 ~/.env.ml 配置后调用 load_api_keys()"
            )
        self._client = openai.OpenAI(api_key=api_key, base_url=_DASHSCOPE_URL)

    def _normalize_image(self, image) -> Image.Image:
        """把 numpy 数组 / PIL 图统一成 PIL 图；超限等比缩小。"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError(f"不支持的图像类型: {type(image)}")
        w, h = image.size
        if w > _SAFEGUARD_IMAGE_RESOLUTION or h > _SAFEGUARD_IMAGE_RESOLUTION:
            image = image.copy()
            image.thumbnail((_SAFEGUARD_IMAGE_RESOLUTION, _SAFEGUARD_IMAGE_RESOLUTION))
        return image

    def ask(self, prompt: str, images=Iterable["Image"]):
        """见 OracleInterface.ask。env.py 里固定传恰好一张目标图。

        返回模型对 `prompt` 的回答字符串（关于目标图的如实答案）。
        """
        image_list = list(images) if images is not None else []
        if len(image_list) > 1:
            raise ValueError(
                f"QwenOracle 一次只支持 1 张图，收到 {len(image_list)} 张"
            )
        if len(prompt) > _SAFEGUARD_N_LETTERS:
            prompt = prompt[:_SAFEGUARD_N_LETTERS]

        content = []
        if image_list:
            image = self._normalize_image(image_list[0])
            fmt = (image.format or "PNG").lower()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{fmt};base64,{_encode_image_b64(image, fmt)}"
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        completion = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=int(self.max_output_length / 4),
            stream=True,
        )
        # 流式拼接最终回答；跳过无内容/推理中间 chunk（与 utils.QwenLLM 一致）
        answer = "".join(
            (chunk.choices[0].delta.content or "")
            for chunk in completion
            if chunk.choices
        )
        return answer.strip()


# ---------------------------------------------------------------------------
# 本地 vLLM oracle（可选）：与 QwenOracle 同接口，走本机 vLLM 的 AWQ 底座。
# 评测脚本注释里说明"本地 oracle 用 AWQ 底座（不要用微调的 qwen32b，它是提问者）"。
# 若你不需要本地 oracle，忽略本类即可。
# ---------------------------------------------------------------------------
class LocalVLLMOracle(OracleInterface):
    """Oracle：用本地 vLLM（ClientBasedLLM）对目标图如实回答。"""

    def __init__(self, model_id: str = None, port: int = 3389):
        from utils import ClientBasedLLM

        self._llm = ClientBasedLLM(model_id=model_id, port=port)

    def ask(self, prompt: str, images=Iterable["Image"]):
        return self._llm.ask(prompt=prompt, images=list(images))
