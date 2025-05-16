import pandas as pd 
# 从原始数据中重新读取，确保一致性
coin20_df = pd.read_csv("github_page/static/primitive_task_skills.csv")
coin50_df = pd.read_csv("github_page/static/task_skills.csv")

# 设置 index
bin_50 = coin50_df.set_index("task_name") > 0
bin_20 = coin20_df.set_index("task_name") > 0

# 技能全集（取并集，不考虑是否有边）
all_skills = sorted(set(bin_20.columns).union(set(bin_50.columns)))

# 添加空节点（左右各一个）
EMPTY_LEFT = "__EMPTY_LEFT__"
EMPTY_RIGHT = "__EMPTY_RIGHT__"

# 三类节点：primitive tasks, skills, interactive tasks, 加上空节点
primitive_tasks = bin_20.index.tolist()
interactive_tasks = bin_50.index.tolist()
all_nodes = [EMPTY_LEFT] + primitive_tasks + all_skills + interactive_tasks + [EMPTY_RIGHT]
node_indices = {name: i for i, name in enumerate(all_nodes)}

# 构建颜色列表
node_colors = (
    ["rgba(0,0,0,0)"] +                 # 左空节点（透明）
    ["#A8D5BA"] * len(primitive_tasks) +  # primitive task
    ["#C8E6C9"] * len(all_skills) +       # skill node
    ["#4CAF50"] * len(interactive_tasks) + # interactive task
    ["rgba(0,0,0,0)"]                  # 右空节点（透明）
)

# 构建边
sources = []
targets = []
values = []

# Part 1: Primitive Task ➝ Skill
for pt in primitive_tasks:
    for skill in all_skills:
        if skill in bin_20.columns and bin_20.loc[pt, skill]:
            sources.append(node_indices[pt])
            targets.append(node_indices[skill])
            values.append(1)

# Part 2: Skill ➝ Interactive Task
for it in interactive_tasks:
    for skill in all_skills:
        if skill in bin_50.columns and bin_50.loc[it, skill]:
            sources.append(node_indices[skill])
            targets.append(node_indices[it])
            values.append(1)

# 识别技能的来源和去向
skills_with_source = set()  # 有primitive task作为来源的技能
skills_with_target = set()  # 有interactive task作为目标的技能

# 检查哪些技能有primitive task作为来源
for pt in primitive_tasks:
    for skill in all_skills:
        if skill in bin_20.columns and bin_20.loc[pt, skill]:
            skills_with_source.add(skill)

# 检查哪些技能有interactive task作为目标
for it in interactive_tasks:
    for skill in all_skills:
        if skill in bin_50.columns and bin_50.loc[it, skill]:
            skills_with_target.add(skill)

# 处理没有来源的技能（只被interactive tasks使用）
for skill in all_skills:
    if skill not in skills_with_source and skill in skills_with_target:
        # 将没有来源的技能连接到左空节点
        sources.append(node_indices[EMPTY_LEFT])
        targets.append(node_indices[skill])
        values.append(0.5)  # 小值，但足以显示

# 处理没有目标的技能（只被primitive tasks使用）
for skill in all_skills:
    if skill in skills_with_source and skill not in skills_with_target:
        # 将没有目标的技能连接到右空节点
        sources.append(node_indices[skill])
        targets.append(node_indices[EMPTY_RIGHT])
        values.append(0.5)  # 小值，但足以显示

# 处理完全孤立的技能（既没有来源也没有去向）
for skill in all_skills:
    if skill not in skills_with_source and skill not in skills_with_target:
        # 将完全孤立的技能连接到左右空节点
        sources.append(node_indices[EMPTY_LEFT])
        targets.append(node_indices[skill])
        values.append(0.3)
        
        sources.append(node_indices[skill])
        targets.append(node_indices[EMPTY_RIGHT])
        values.append(0.3)

# 创建图
import plotly.graph_objects as go
fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=10,
        thickness=12,
        line=dict(color="gray", width=0.5),
        label=all_nodes,
        color=node_colors
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=["rgba(102,187,106,0.35)" if v > 0.05 else "rgba(0,0,0,0.05)" for v in values]
    ))])

# 设置图表尺寸为 20:10 的长宽比
fig.update_layout(
    title_text="Full Three-Stage Skill Mapping: Primitive ➝ Skill ➝ Interactive", 
    font_size=10,
    width=2000,  # 宽度设置为 2000 像素
    height=1000,  # 高度设置为 1000 像素
    margin=dict(l=20, r=20, t=50, b=20),  # 调整边距以提供更多空间
    paper_bgcolor='rgba(0,0,0,0)',  # 透明背景
    plot_bgcolor='rgba(0,0,0,0)'    # 透明背景
)

# 保存图
full_skill_sankey_path = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/github_page/static/plots/plot_skill_mapping.html"
fig.write_html(full_skill_sankey_path)


