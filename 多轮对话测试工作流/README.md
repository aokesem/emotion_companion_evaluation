# 多轮对话测试工作流

当前目标：运行模拟用户 AI 与被测 AI 的多轮精神慰藉对话，并保存主观评分与评估结果。

## 流程

第一版固定每个用户 `10` 轮对话，一轮定义为：

```text
模拟用户一句 + 被测 AI 一句
```

成功样本完整流程：

```text
用户 AI 开场
-> 被测 AI 回复
-> 用户 AI 回复
-> 循环至固定轮数
-> 用户 AI 主观评分
-> 被测 AI 事后总结
-> 评估 AI 客观评分
-> 保存成功结果
```

如果中途任一节点失败，不保存成功结果，只写入错误记录。下次运行时该 ID 仍可再次被抽中。

## 文件

- `workflow_config.json`：默认路径、轮数、运行参数、模拟用户/评估 AI 模型配置
- `tested_model_profiles.json`：被测模型 profile 配置
- `run_langgraph_workflow.py`：LangGraph 主流程脚本
- `extract_eval_scores.py`：从评分摘要中提取精简评分表
- `prompts/模拟用户对话提示词.txt`：模拟用户对话提示词
- `prompts/被测AI对话提示词.txt`：被测 AI 对话提示词
- `prompts/模拟用户评分提示词.txt`：模拟用户主观评分提示词
- `prompts/被测AI总结提示词.txt`：被测 AI 事后总结提示词
- `prompts/评估AI评分提示词.txt`：评估 AI 客观评分提示词

## 输出

- `outputs/dialog_eval_runs.jsonl`：完整成功结果
- `outputs/dialog_eval_summary.csv`：评分摘要
- `outputs/dialog_eval_errors.jsonl`：错误记录

可通过 `--output-dir` 指定实验目录。相对路径默认位于 `outputs/` 下，例如：

```powershell
python .\多轮对话测试工作流\run_langgraph_workflow.py --output-dir deepseek-v4-pro-v1 --limit 10 --continue-on-error
```

此时输出会写入：

```text
多轮对话测试工作流/outputs/deepseek-v4-pro-v1/
```


## 被测模型 Profile

被测模型配置集中放在：

```text
多轮对话测试工作流/tested_model_profiles.json
```

`workflow_config.json` 中的 `tested_profile` 决定默认被测模型。运行时也可以用 `--tested-profile` 临时指定：

```powershell
python .\多轮对话测试工作流\run_langgraph_workflow.py --tested-profile zhipu_GLM4.7flash --limit 1 --print-dialog --continue-on-error
```

指定 profile 后，脚本会自动读取该 profile 的：

- `base_url` 或 `base_url_env`
- `api_key_env`
- `model`
- `output_dir`
- `auto_append_v1`
- `chat_completions_path`

输出目录默认使用 profile 中的 `output_dir`，例如 `zhipu_GLM4.7flash` 会写入：

```text
多轮对话测试工作流/outputs/zhipu_GLM4.7flash/
```

如果你额外传了 `--output-dir`，则以命令行指定的目录为准。

API key 不建议写进 profile 文件。请在 `.env` 中写对应变量，例如：

```env
ZHIPU_API_KEY=你的智谱key
```

智谱这类不使用 `/v1/chat/completions` 的接口，可以在 profile 中设置：

```json
"auto_append_v1": false,
"chat_completions_path": "/chat/completions"
```
## Dry Run

检查数据匹配、成功记录跳过状态、提示词文件是否为空，不调用模型：

```powershell
python .\多轮对话测试工作流\run_langgraph_workflow.py --dry-run --start 0 --limit 5
```

## 正式运行

提示词补齐后可运行：

```powershell
python .\多轮对话测试工作流\run_langgraph_workflow.py --start 0 --limit 1 --turns 10 --continue-on-error
```

按指定 ID 运行：

```powershell
python .\多轮对话测试工作流\run_langgraph_workflow.py --ids 14643113001 --turns 10 --continue-on-error
```

脚本会读取本目录下 `.env`，如果不存在，则继续尝试读取 `../数据生成代码/.env`。

## 手动被测模式

使用 `--manual` 可以手动输入每轮被测 AI 回复。该模式固定完成设定轮数，默认输出到 `outputs/manual/`，只生成模拟用户主观评分，跳过被测 AI 事后总结和评估 AI 评分。

```powershell
python .\多轮对话测试工作流\run_langgraph_workflow.py --manual --ids 14643113001 --turns 10
```

## 提取评分

默认从 `outputs/dialog_eval_summary.csv` 生成 `outputs/dialog_eval_scores.csv`：

```powershell
python .\多轮对话测试工作流\extract_eval_scores.py
```

指定实验目录：

```powershell
python .\多轮对话测试工作流\extract_eval_scores.py --output-dir deepseek-v4-pro-v1
```

