# 事实数据的筛选过程
1.原始CHARLS数据
原始数据位于CHARLS2020r文件夹下，综合概览数据集之后，将认为有价值的列及相应信息存储到 \用户画像数据\charls数据选择表.csv下作为后续数据处理依据。
实际使用涉及以下六张表
- Sample_Infor.dta
- Weights.dta
- Demographic_Background.dta
- Health_Status_and_Functioning.dta
- Work_Retirement.dta
- Family_Information.dta

2.提取画像字段
按字段目录，使用 \数据处理代码\筛选用户画像列.py，根据个体键：ID、householdID 与 家庭键：householdID 从上述六张 DTA 保留画像相关字段和合并键。
3.合并为个人级宽表
使用 \数据处理代码\合并用户画像表.py 以 Sample_Infor.dta 为骨架，保留crosssection = 1，died = 0条件（即 2020 截面、可进行全国推断且健在的老人）。
然后Weights / Demographic / Health / Work按 ID 左连接；Family_Information按 householdID 左连接，得到 \个人级宽表 用户画像_合并.dta。
4.筛为分析样本
使用 数据处理代码\筛选用户画像合并表.py 在宽表上继续筛选：
- crosssection = 1
- died = 0
- INDV_weight_ad2 非缺失且 > 0
- proxy_2 != 1
- proxy_5 != 1
此步剔除代答记录，保留有有效个人权重、可由本人信息构成画像的样本。\整理后数据\用户画像_分析样本.dta
5.年龄筛选与加权抽样
使用 \数据处理代码\抽取用户画像样本.py 读取分析样本后：先筛 xrage >= 45，再以 INDV_weight_ad2 归一化为抽样概率，按该概率有放回抽取 500 次。
默认抽取种子为20260602，生成 \用户画像数据\用户画像_抽样500.dta与 \用户画像数据\用户画像_抽样500.csv
6.生成事实文本
使用 \数据处理代码\生成用户画像事实文本.py 读取 \用户画像数据\用户画像_抽样500.csv，把每一行编码数据映射为中文的半结构化事实。
分组依据是/用户画像数据/用户画像信息分组.csv，共九组，分为
- 基础信息
- 教育信息
- 婚姻信息
- 情绪信息
- 健康信息
- 活动信息
- 上网信息
- 子女信息
- 工作信息
脚本会处理：CHARLS 缺失码，如 97/98/99/997/998/999；性别、城乡、学历、婚姻、生活满意度等分类编码；15 类慢性病字段；体力活动、社会活动、互联网使用；前两名子女的信息与联系频率；农业劳动、退休、工作天数等。
最终得到用户画像数据\用户画像_抽样500_事实文本.csv

# 文件说明
在多轮对话测试工作流的workflow_config.json保存了全局设置，例如轮数、默认处理条数、模拟用户模型、评估模型、数据路径、prompt 路径，默认模拟用户模型和评估模型为deepseek v4 pro。

tested_model_profiles.json中保存“被测模型”的配置。修改文件或通过命令行选择profile来更改被测模型的配置

.env 环境变量保存 API key 等敏感信息，以供其他文件读取

extract_eval_scores.py，运行该文件，从结果中提取精简评分表和平均分。

# 运行方法
使用以下命令开始对被测模型测试：
python .\多轮对话测试工作流\run_langgraph_workflow.py --tested-profile glm-4-plus  --start 0 --limit 1 --turns 10 --print-dialog --continue-on-error      
--tested-profile zhipu_GLM4.7flash：选择被测模型名称，模型名称可在tested_model_profiles.json中定义。
--start 0：从样本第 0 行开始。
--limit 1：只跑 1 个用户。
--turns 10：对话 10 轮（不推荐更改）
--print-dialog：在终端实时打印出对话。
--continue-on-error：在某条报错后不停止，继续运行。

每个 profile 会自动输出到自己的目录，在完成运行后可以使用python .\多轮对话测试工作流\extract_eval_scores.py --output-dir deepseek_v4_pro自动提取指定目录（output-dir）下的评分并生成文件。

使用 --manual 进行手动模式。手动模式下模拟用户 AI 正常发言，需要在终端手动输入被测 AI 回复。默认输出到outputs/manual/。只生成模拟用户 AI 主观评分，不生成被测 AI 总结，不生成评估 AI 评分。