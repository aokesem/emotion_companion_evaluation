"""
读取 CHARLS2020r/Sample_Infor.dta，只打印第一行（第 1 个受访者）。
运行: python 打印数据集的第一行.py
"""

import pandas as pd

path = r"d:\代码_精神慰藉agent评估\CHARLS2020r\Demographic_Background.dta"

df = pd.read_stata(path)

first = df.iloc[0]

print("第 1 行（index=0）：\n")
for col in df.columns:
    print(f"  {col}: {first[col]}")
