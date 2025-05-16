import plotly.graph_objects as go

models = [
    "Voxposer_mean",
    "Voxposer_topdown",
    "Manigen_fake",
    "Rekep_fake",
]
avg_sr = [0.1063, 0.1125, 0.5, 0.3]

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=models,
        y=avg_sr,
        marker=dict(
            color="orange",
            line=dict(color="darkorange", width=1),
        ),
        text=[f"{v:.2f}" for v in avg_sr],
        textposition="outside",
        hoverinfo="x+y",
        width=0.5,  # helps simulate some "roundness"
    )
)

# Rounded corners: Simulated with layout tweaks
fig.update_traces(marker_line_width=0, opacity=0.9)

fig.update_layout(
    title="Benchmark Model Success Rates",
    yaxis=dict(range=[0, max(avg_sr)], title="Average Success Rate"),
    xaxis=dict(title="Model"),
    plot_bgcolor="white",
    bargap=0.3,
    showlegend=False,
)

# fig.show()
# fig.write_image(
#     "evaluation_results/code_as_policy_benchmark_success_rate.png",
#     width=1200,
#     height=800,
#     scale=2,
# )
fig.write_image(
    "evaluation_results/code_as_policy_benchmark_success_rate.svg",
)
