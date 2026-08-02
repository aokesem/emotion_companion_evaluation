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

可以在这里调整：

- 默认模型名
- temperature / max_tokens / timeout / retries / sleep
- system prompt
- 四步 prompt 模板
- 每一步的必需输出字段


## 3. 运行
默认读取：
`用户画像数据/用户画像_抽样500_事实文本.csv`

```powershell
python .\数据生成代码\generate_sim_user_pipeline.py --model deepseek-v4-pro --continue-on-error
```

## 4. 常用参数

- `--start 0 --limit 20`：先跑前 20 条做抽查
- 默认会按已写入 JSONL 的 `ID` 跳过已有输出，避免重复生成同一用户
- `--regenerate-existing`：不跳过已有输出，强制重新生成
- `--resume`：兼容旧参数；当前默认已经跳过已有输出
- `--sleep 0.2`：每条之间间隔，降低速率
- `--temperature 0.4`：控制生成稳定性和多样性
- `--max-tokens 1200`：每步输出上限

说明：如果输入文件中同一个 `ID` 出现多次，脚本只处理第一次出现的记录，后续重复 `ID` 会直接跳过；如果输出 JSONL 中已经存在某个 `ID`，默认也会跳过。

示例：

```powershell
python .\数据生成代码\generate_sim_user_pipeline.py --model deepseek-v4-pro --start 0 --limit 20 --continue-on-error
python .\数据生成代码\generate_sim_user_pipeline.py --model deepseek-v4-pro --continue-on-error
python .\数据生成代码\generate_sim_user_pipeline.py --model deepseek-v4-pro --regenerate-existing --start 0 --limit 5 --continue-on-error
```

## 5. 输出文件
在 `用户画像数据/` 下生成：

- `模拟用户信息_500.jsonl`：逐条完整结构
- `模拟用户信息_500.csv`：扁平化汇总
- `模拟用户信息_500_errors.jsonl`：失败记录

## 6. 四步串行逻辑

1. 用户性格（性格偏好、对话风格偏好）
2. 个人经历（6项）
3. 生活烦恼（主要烦恼、烦恼类别、表层话题、深层成因）
4. 对话锚点（情绪状态、触发事件、开场方式、开场首句示例）


