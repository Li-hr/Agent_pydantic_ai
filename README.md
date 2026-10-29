# 简单图像 Agent 示例

最小可运行示例：一个 Agent 文件 + 一个工具文件，演示 pydantic-ai 挂载工具与多轮对话。

## 文件

| 文件 | 说明 |
|------|------|
| `simple_image_agent.py` | Agent 定义、工具注册、会话状态、交互入口 |
| `simple_image_tools.py` | 纯函数工具：图像裁剪、灰度阈值分割 |

## 前置条件

- 在项目根目录 `config/.env` 配置 DeepSeek API Key（见 `config/env.example`）
- 依赖：项目已有的 `pydantic-ai`、`PIL`、`numpy`

## 用法

在项目根目录执行：

```bash
# 多轮交互（保持对话上下文）
python examples/simple_image_agent.py

# 单轮
python examples/simple_image_agent.py "把 D:/img.png 裁成 0,0-200,200，再阈值 128 分割"
```

交互命令：

- `reset` — 清空对话历史与最近图像路径
- `exit` / `quit` — 退出

## 可用工具

1. **crop_image** — 按像素框 `(left, top, right, bottom)` 裁剪
2. **threshold_segment** — 灰度阈值二值分割（默认 threshold=128）

## 会话上下文

`ImageAgentDeps` 跨轮保留：

- `history`：`message_history`，多轮对话
- `last_input_path` / `last_output_path`：最近处理的图像，便于「对刚才那张再分割」这类指令
