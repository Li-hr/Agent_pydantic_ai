# -*- coding: utf-8 -*-
"""
最小图像 Agent 示例（仅两个工具：裁剪 + 阈值分割）。
支持多轮持久对话：message_history 跨轮保留。

用法（在项目根目录）:
  # 单轮
  python examples/simple_image_agent.py "把 D:/img.png 裁成 0,0-200,200"
  # 多轮交互
  python examples/simple_image_agent.py
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage

# 允许从任意 cwd 直接运行本文件
_ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.models import deepseek_model  # noqa: E402
from simple_image_tools import crop_image, threshold_segment  # noqa: E402


SYSTEM_PROMPT = """你是一个简单的图像处理助手。
可用工具：
1) crop_image — 按像素框裁剪
2) threshold_segment — 灰度阈值二值分割

规则：
- 可使用会话上下文中的上一张输入/输出路径，用户说「刚才那张」「再分割一次」时优先复用
- 需要时可以串联：先裁剪，再对裁剪结果做阈值分割
- 最终用中文简要汇报输出文件路径与关键参数
"""


@dataclass
class ImageAgentDeps:
    """会话状态：对话历史 + 最近图像路径。"""

    history: List[ModelMessage] = field(default_factory=list)
    last_input_path: Optional[str] = None
    last_output_path: Optional[str] = None
    max_history: int = 60


simple_image_agent = Agent(
    deepseek_model,
    deps_type=ImageAgentDeps,
    system_prompt=SYSTEM_PROMPT,
)


@simple_image_agent.instructions
def inject_session_context(ctx: RunContext[ImageAgentDeps]) -> str:
    deps = ctx.deps
    lines = ["【会话上下文】"]
    if deps.last_input_path:
        lines.append(f"- 最近输入图: {deps.last_input_path}")
    if deps.last_output_path:
        lines.append(f"- 最近输出图: {deps.last_output_path}")
    if len(lines) == 1:
        lines.append("- 尚无图像记录，请用户提供路径")
    return "\n".join(lines)


class CropParams(BaseModel):
    image_path: str = Field(..., description="输入图像路径")
    left: int = Field(..., description="裁剪左边界 (px)")
    top: int = Field(..., description="裁剪上边界 (px)")
    right: int = Field(..., description="裁剪右边界 (px，不含)")
    bottom: int = Field(..., description="裁剪下边界 (px，不含)")
    output_path: Optional[str] = Field(None, description="可选输出路径")


class ThresholdParams(BaseModel):
    image_path: str = Field(..., description="输入图像路径")
    threshold: int = Field(128, description="灰度阈值 0–255")
    invert: bool = Field(False, description="True 时取低于阈值为前景")
    output_path: Optional[str] = Field(None, description="可选输出路径")


@simple_image_agent.tool
def crop_image_tool(ctx: RunContext[ImageAgentDeps], params: CropParams) -> str:
    """按矩形框裁剪图像并保存。"""
    out = crop_image(
        params.image_path,
        params.left,
        params.top,
        params.right,
        params.bottom,
        params.output_path,
    )
    ctx.deps.last_input_path = params.image_path
    ctx.deps.last_output_path = out
    return f"裁剪完成: {out}"


@simple_image_agent.tool
def threshold_segment_tool(ctx: RunContext[ImageAgentDeps], params: ThresholdParams) -> str:
    """对图像做灰度阈值分割并保存二值图。"""
    out = threshold_segment(
        params.image_path,
        threshold=params.threshold,
        invert=params.invert,
        output_path=params.output_path,
    )
    ctx.deps.last_input_path = params.image_path
    ctx.deps.last_output_path = out
    return f"阈值分割完成: {out}"


def _trim_history(deps: ImageAgentDeps) -> None:
    if len(deps.history) > deps.max_history:
        deps.history = deps.history[-deps.max_history :]


async def run_simple_image_task(
    task: str,
    deps: Optional[ImageAgentDeps] = None,
) -> str:
    """单轮调用；传入同一 deps 即可跨轮保持对话与图像路径。"""
    if deps is None:
        deps = ImageAgentDeps()
    result = await simple_image_agent.run(
        task,
        deps=deps,
        message_history=deps.history or None,
    )
    deps.history = result.all_messages()
    _trim_history(deps)
    return result.output


async def chat_loop(deps: Optional[ImageAgentDeps] = None) -> None:
    """交互式多轮对话；输入 empty / exit / quit 结束，reset 清空上下文。"""
    deps = deps or ImageAgentDeps()
    print("简单图像 Agent（多轮）。命令: reset=清空上下文, exit/quit=退出")
    while True:
        try:
            user = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出")
            break
        if not user:
            continue
        low = user.lower()
        if low in {"exit", "quit", "q"}:
            print("已退出")
            break
        if low == "reset":
            deps.history.clear()
            deps.last_input_path = None
            deps.last_output_path = None
            print("上下文已清空")
            continue
        reply = await run_simple_image_task(user, deps=deps)
        print(f"Agent> {reply}")


def main() -> None:
    parser = argparse.ArgumentParser(description="简单图像 Agent（支持多轮上下文）")
    parser.add_argument("task", nargs="?", default=None, help="单轮任务；省略则进入交互模式")
    args = parser.parse_args()
    if args.task:
        print(asyncio.run(run_simple_image_task(args.task)))
    else:
        asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
