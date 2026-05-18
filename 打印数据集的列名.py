"""
读取 CHARLS2020r/Sample_Infor.dta，只打印列名（变量名）。
运行: python print_sample_infor_columns.py
"""

import pandas as pd

# 按你的实际路径修改
path = r"d:\代码_精神慰藉agent评估\CHARLS2020r\Sample_Infor.dta"

df = pd.read_stata(path)

print(f"共 {len(df.columns)} 列：\n")
for i, name in enumerate(df.columns, start=1):
    print(f"{i:2d}. {name}")
