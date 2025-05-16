import plotly.graph_objects as go

# 准备桑基图数据
source_skills = list(nonzero_matrix.index)
target_skills = list(nonzero_matrix.columns)

# 创建节点列表（合并 source 和 target）
all_nodes = source_skills + target_skills
node_indices = {skill: idx for idx, skill in enumerate(all_nodes)}

# 构建连接关系
sources = []
targets = []
values = []

for source in source_skills:
    for target in target_skills:
        weight = nonzero_matrix.loc[source, target]
        if weight > 0:
            sources.append(node_indices[source])
            targets.append(node_indices[target])
            values.append(weight)

# 画桑基图
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="gray", width=0.5),
        label=all_nodes,
        color=["#A8D5BA"] * len(all_nodes)
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=["rgba(102,187,106,0.4)"] * len(sources)
    ))])

fig.update_layout(title_text="COIN-50 ➝ COIN-20 Skill Mapping (Sankey Diagram)", font_size=10)
fig.show()
