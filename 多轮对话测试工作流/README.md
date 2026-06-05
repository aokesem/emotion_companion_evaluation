# 多轮对话测试工作流

当前目标：搭建 LangGraph 框架，不先编写具体提示词。

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

- `workflow_config.json`：默认路径、模型、轮数和运行参数
- `run_langgraph_workflow.py`：LangGraph 主流程脚本
- `prompts/simulated_user_dialog.txt`：模拟用户对话提示词，暂为空
- `prompts/tested_agent_dialog.txt`：被测 AI 对话提示词，暂为空
- `prompts/simulated_user_rating.txt`：模拟用户主观评分提示词，暂为空
- `prompts/tested_agent_summary.txt`：被测 AI 事后总结提示词，暂为空
- `prompts/evaluator_rating.txt`：评估 AI 客观评分提示词，暂为空

## 输出

- `outputs/dialog_eval_runs.jsonl`：完整成功结果
- `outputs/dialog_eval_summary.csv`：评分摘要
- `outputs/dialog_eval_errors.jsonl`：错误记录

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
