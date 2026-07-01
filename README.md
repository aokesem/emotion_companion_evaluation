python .\多轮对话测试工作流\run_langgraph_workflow.py --start 0 --limit 1 --turns 10 --print-dialog --continue-on-error --output-dir zhipu_GLM4.7flash


#文件说明
在多轮对话测试工作流的workflow_config.json保存了全局设置，例如轮数、默认处理条数、模拟用户模型、评估模型、数据路径、prompt 路径，默认模拟用户模型和评估模型为deepseek v4 pro。

tested_model_profiles.json中保存“被测模型”的配置。修改文件或通过命令行选择profile来更改被测模型的配置

.env 环境变量保存 API key 等敏感信息，以供其他文件读取

extract_eval_scores.py，运行该文件，从结果中提取精简评分表和平均分。

#运行方法
使用一下命令开始对被测模型测试：
python .\多轮对话测试工作流\run_langgraph_workflow.py --tested-profile zhipu_GLM4.7flash --start 0 --limit 1 --turns 10 --print-dialog --continue-on-error
参数含义：
--tested-profile zhipu_GLM4.7flash：选择被测模型名称，模型名称可在tested_model_profiles.json中定义。
--start 0：从样本第 0 行开始。
--limit 1：只跑 1 个用户。
--turns 10：对话 10 轮（不推荐更改）
--print-dialog：在终端实时打印出对话。
--continue-on-error：在某条报错后不停止，继续运行。

每个 profile 会自动输出到自己的目录，在完成运行后可以使用python .\多轮对话测试工作流\extract_eval_scores.py --output-dir deepseek_v4_pro自动提取指定目录（output-dir）下的评分并生成文件。

使用 --manual 进行手动模式。手动模式下模拟用户 AI 正常发言，需要在终端手动输入被测 AI 回复。默认输出到outputs/manual/。只生成模拟用户 AI 主观评分，不生成被测 AI 总结，不生成评估 AI 评分。