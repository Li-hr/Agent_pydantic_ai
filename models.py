# -*- coding: utf-8 -*-
"""共享 LLM 模型配置（文本 DeepSeek + 可选 Qwen 视觉）"""
from __future__ import annotations

import os
from typing import Optional

import httpx
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from core.paths import CONFIG_DIR, PROJECT_ROOT

load_dotenv(CONFIG_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

### 全局模型共享 ###
DEEPSEEK_API_KEY = os.getenv("deepseek_api_key") or os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("未找到 deepseek_api_key，请在 config/.env 中配置")

# Windows 系统代理常导致 DeepSeek TLS EOF；默认不走环境代理。
# 若必须走代理，在 config/.env 设 DEEPSEEK_TRUST_ENV=1
_TRUST_ENV = os.getenv("DEEPSEEK_TRUST_ENV", "").strip().lower() in {"1", "true", "yes"}
_http_client = httpx.AsyncClient(trust_env=_TRUST_ENV, timeout=httpx.Timeout(120.0, connect=30.0))

# API 现仅支持 deepseek-v4-flash / deepseek-v4-pro（旧名 deepseek-chat 已失效）
DEEPSEEK_MODEL = (
    os.getenv("DEEPSEEK_MODEL") or os.getenv("deepseek_model") or "deepseek-v4-flash"
).strip()

deepseek_model = OpenAIChatModel(
    DEEPSEEK_MODEL,
    provider=OpenAIProvider(
        base_url="https://api.deepseek.com/v1",
        api_key=DEEPSEEK_API_KEY,
        http_client=_http_client,
    ),
)

# ---------------------------------------------------------------------------
# Qwen 视觉（可选）：未配置 key 时 vision 工具不可用，不影响文本 Agent
# 百炼 DashScope OpenAI 兼容：https://dashscope.aliyuncs.com/compatible-mode/v1
# ---------------------------------------------------------------------------
QWEN_API_KEY = (
    os.getenv("QWEN_API_KEY")
    or os.getenv("qwen_api_key")
    or os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("dashscope_api_key")
    or ""
).strip()

QWEN_VL_MODEL = (
    os.getenv("QWEN_VL_MODEL") or os.getenv("qwen_vl_model") or "qwen-vl-max"
).strip()

QWEN_VL_BASE_URL = (
    os.getenv("QWEN_VL_BASE_URL")
    or os.getenv("qwen_vl_base_url")
    # 国内北京默认；新加坡/国际请在 .env 覆盖
    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
).strip()

_qwen_http_client = httpx.AsyncClient(
    trust_env=_TRUST_ENV,
    timeout=httpx.Timeout(180.0, connect=30.0),
)

qwen_vision_model: Optional[OpenAIChatModel] = None
if QWEN_API_KEY:
    qwen_vision_model = OpenAIChatModel(
        QWEN_VL_MODEL,
        provider=OpenAIProvider(
            base_url=QWEN_VL_BASE_URL,
            api_key=QWEN_API_KEY,
            http_client=_qwen_http_client,
        ),
    )


def is_vision_configured() -> bool:
    return qwen_vision_model is not None


def get_qwen_vision_model() -> OpenAIChatModel:
    if qwen_vision_model is None:
        raise RuntimeError(
            "未配置 Qwen 视觉模型。请在 config/.env 填写 QWEN_API_KEY "
            "（或 DASHSCOPE_API_KEY），可选 QWEN_VL_MODEL / QWEN_VL_BASE_URL。"
        )
    return qwen_vision_model
