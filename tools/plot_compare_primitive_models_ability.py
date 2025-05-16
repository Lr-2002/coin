import plotly.express as px
import pandas as pd
import os
import plotly.io as pio

# 替换为你的数据（两行）
average_score = [16.67, 10.48, 19.05, 5.71, 19.52, 18.57]
non_zero_ability = [12, 4, 10, 3, 5, 7]
model_names = [f"Model {chr(65+i)}" for i in range(len(average_score))]
model_names = [
    "Gr00t(120000)",
    "Pi0",
    "CogACT",
    "Rekep",
    "Voxposer_normal",
    "Voxposer_topdown",
]

# 构建 DataFrame
df = pd.DataFrame(
    {
        "Model": model_names,
        "Average Score (%)": average_score,
        "Non-zero Ability": non_zero_ability,
    }
)

# 绘制二维散点图
fig = px.scatter(
    df,
    x="Non-zero Ability",
    y="Average Score (%)",
    text="Model",
    title="Model Capability Landscape",
    labels={
        "Non-zero Ability": "Task Coverage (Non-zero Abilities)",
        "Average Score (%)": "Average Task Score (%)",
    },
)

fig.update_traces(
    textposition="top center",
    marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey")),
)

fig.update_layout(title_x=0.5)

# 创建输出目录（如果不存在）
output_dir = os.path.join('/home/lr-2002/project/reasoning_manipulation/ManiSkill/github_page/static')
os.makedirs(output_dir, exist_ok=True)

# 保存图表为HTML文件（交互式）
html_path = os.path.join(output_dir, 'model_capability_landscape.html')
fig.write_html(html_path)

# 保存图表为PNG图像（静态）
png_path = os.path.join(output_dir, 'model_capability_landscape.png')
fig.write_image(png_path, width=1000, height=600, scale=2)

# 保存图表为PDF（矢量格式，适合出版）
pdf_path = os.path.join(output_dir, 'model_capability_landscape.pdf')
fig.write_image(pdf_path, width=1000, height=600, scale=2)

print(f"Plot saved as HTML: {html_path}")
print(f"Plot saved as PNG: {png_path}")
print(f"Plot saved as PDF: {pdf_path}")

# 显示图表（如果在交互环境中运行）
fig.show()
