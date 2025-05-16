import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# === STEP 1: 读取 CSV 文件 ===
csv_path = "primitive_csv_from_excel.csv"  # ← 修改为你的文件路径
df_raw = pd.read_csv(csv_path)

# === STEP 2: 预处理 ===
print(df_raw)
# 清洗任务名
df_raw["Task"] = df_raw["Task"].apply(lambda x: re.sub(r"^Tabletop-", "", x))
df_raw["Task"] = df_raw["Task"].apply(lambda x: re.sub(r"-v1$", "", x))

# 设置任务为 index
df_raw.set_index("Task", inplace=True)

# 去掉百分号，转 float
for col in df_raw.columns:
    df_raw[col] = (
        df_raw[col].astype(str).str.replace("%", "", regex=False).astype(float)
    )


# 清洗模型列名
def simplify_column(col):
    col = col.replace("_normal", "(N)").replace("_topdown", "(TD)")
    col = re.sub(r"_[0-9]{8}_[0-9]{6}", "", col)  # 删除时间戳
    return col


df_raw.columns = [simplify_column(c) for c in df_raw.columns]

# === STEP 3: 转置 DataFrame ===
df_t = df_raw.transpose()

# === STEP 4: 绘图 ===
plt.figure(figsize=(14, 5))  # 宽一些，高度压扁
sns.heatmap(df_t, cmap="Greens", linewidths=0.5, annot=True, fmt=".0f", cbar=True)
plt.xticks(rotation=45, ha="right", fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("heatmap_transposed.png", dpi=300)
plt.show()
