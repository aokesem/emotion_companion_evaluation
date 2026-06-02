# 模拟用户信息生成 Pipeline

## 1. 配置 `.env`
先在 `数据生成代码` 目录下复制一份：

```powershell
Copy-Item .\数据生成代码\.env.example .\数据生成代码\.env
```

然后编辑 `.env`：

```env
OPENAI_BASE_URL=https://your-api-host
OPENAI_API_KEY=your-api-key
```

脚本启动时会优先读取同目录下的 `.env`。

> 脚本会自动补全 `/v1` 路径（如果你没写）。

## 2. Prompt 和模型参数
统一放在：

`数据生成代码/pipeline_config.json`

你可以在这里调整：

- 默认模型名
- temperature / max_tokens / timeout / retries / sleep
- system prompt
- 四步 prompt 模板
- 每一步的必需输出字段

主脚本只负责执行，不再把 prompt 硬编码在代码里。

## 3. 运行
默认读取：
`用户画像数据/用户画像_抽样1000_事实文本.csv`

```powershell
python .\数据生成代码\generate_sim_user_pipeline.py --model deepseek-v4-pro --continue-on-error
```

## 4. 常用参数

- `--start 0 --limit 20`：先跑前 20 条做抽查
- `--resume`：断点续跑（按已写入 JSONL 的 ID 跳过）
- `--sleep 0.2`：每条之间间隔，降低速率
- `--temperature 0.2`：控制生成稳定性
- `--max-tokens 1200`：每步输出上限

示例：

```powershell
python .\数据生成代码\generate_sim_user_pipeline.py --model deepseek-v4-pro --start 0 --limit 20 --continue-on-error
python .\数据生成代码\generate_sim_user_pipeline.py --model deepseek-v4-pro --resume --continue-on-error
```

## 5. 输出文件
在 `用户画像数据/` 下生成：

- `模拟用户信息_1000.jsonl`：逐条完整结构
- `模拟用户信息_1000.csv`：扁平化汇总
- `模拟用户信息_1000_errors.jsonl`：失败记录

## 6. 四步串行逻辑

1. 用户性格（性格偏好、对话风格偏好）
2. 个人经历（6项）
3. 生活烦恼（主要烦恼、表层话题、深层话题）
4. 对话锚点（情绪状态、触发事件、开场方式、开场首句示例）

每一步都会引用上一步结果，符合你定义的依赖链。
