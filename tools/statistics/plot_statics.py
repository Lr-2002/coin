import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import os
from matplotlib import rcParams

# 配色集中管理
COLORS = {
    "main": "#8BC99E",      # 主体色（微深绿色）
    "edge": "#4CAF50",      # 条形边框色
    "bg": "none",          # 背景色（透明）
    "coin20": "#A8D5BA",    # COIN-20 颜色
    "coin50": "#4CAF50",    # COIN-50 颜色
}

# 加载数据
def load_data():
    # 读取技能分布数据
    primitive_skill_dist = pd.read_csv("github_page/static/primitive_skill_distribution.csv")
    interactive_skill_dist = pd.read_csv("github_page/static/skill_distribution.csv")
    
    # 读取任务长度数据
    # 模拟数据，实际使用时请替换为真实数据
    task_lengths = [11, 38, 35, 16]  # 任务长度分布
    task_labels = ['0-100', '100-250', '250-500', '500+']  # 对应标签
    
    # 读取子任务长度数据
    subtask_counts = [7, 13, 20, 27, 33]  # 子任务长度分布
    subtask_labels = ['1', '2', '3', '4', '5']  # 对应标签
    
    return {
        "primitive_skills": dict(zip(primitive_skill_dist['skill'], primitive_skill_dist['count'])),
        "interactive_skills": dict(zip(interactive_skill_dist['skill'], interactive_skill_dist['count'])),
        "task_lengths": task_lengths,
        "task_labels": task_labels,
        "subtask_counts": subtask_counts,
        "subtask_labels": subtask_labels
    }

# 创建单个图表并保存
def create_single_plot(plot_type, data, save_dir=None):
    # 设置样式
    sns.set_style("whitegrid")
    rcParams['axes.facecolor'] = 'none'
    rcParams['figure.facecolor'] = 'none'
    rcParams['font.size'] = 11  # Set base font size to 11
    
    # 创建图表
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # 添加更圆滑的背景
    rect = patches.FancyBboxPatch(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        boxstyle="round,pad=0.15",  # 增大padding使圆角更明显
        linewidth=0,
        facecolor=COLORS["bg"],
        zorder=-1
    )
    ax.add_patch(rect)
    
    # 根据图表类型绘制不同的图表
    filename = ""
    if plot_type == "coin20_skills":
        primitive_skills = data["primitive_skills"]
        sorted_skills = dict(sorted(primitive_skills.items(), key=lambda x: x[1], reverse=True))
        plt.bar(list(sorted_skills.keys()), list(sorted_skills.values()), color=COLORS["coin20"], edgecolor=COLORS["edge"], width=0.7)
        plt.title('COIN-20 Skill Distribution', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.xlabel('Skills', fontsize=14, labelpad=10)
        plt.ylabel('Number of Tasks', fontsize=14, labelpad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout(pad=2.0)
        filename = "coin20_skill_distribution.png"
        
    elif plot_type == "coin50_skills":
        interactive_skills = data["interactive_skills"]
        sorted_skills = dict(sorted(interactive_skills.items(), key=lambda x: x[1], reverse=True))
        plt.bar(list(sorted_skills.keys()), list(sorted_skills.values()), color=COLORS["coin50"], edgecolor=COLORS["edge"], width=0.7)
        plt.title('COIN-50 Skill Distribution', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.xlabel('Skills', fontsize=14, labelpad=10)
        plt.ylabel('Number of Tasks', fontsize=14, labelpad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout(pad=2.0)
        filename = "coin50_skill_distribution.png"
        
    elif plot_type == "task_length":
        result = plt.pie(data["task_lengths"], labels=data["task_labels"], colors=[COLORS["coin50"]]*len(data["task_lengths"]),
               autopct='%1.1f%%', startangle=140, textprops={'fontsize': 13}, 
               wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        wedges = result[0]
        plt.title('COIN-50 Task Length Distribution', fontsize=16, pad=20)
        
        # Add legend
        # plt.legend(wedges, [f"{label} steps" for label in data["task_labels"]], 
        #           title="Steps Ranges", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.tight_layout(pad=2.0)
        filename = "coin50_task_length_distribution.png"
        
    elif plot_type == "subtask_count":
        result = plt.pie(data["subtask_counts"], labels=data["subtask_labels"], colors=[COLORS["coin50"]]*len(data["subtask_counts"]),
               autopct='%1.1f%%', startangle=140, textprops={'fontsize': 13},
               wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        wedges = result[0]
        plt.title('COIN-50 Subtask Length Distribution', fontsize=16, pad=20)
        
        # Add legend
        # plt.legend(wedges, [f"{label} steps" for label in data["subtask_labels"]], 
        #           title="Number of Steps", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.tight_layout(pad=2.0)
        filename = "coin50_subtask_count_distribution.png"
    
    # 保存图表
    if save_dir and filename:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.3)
        print(f"Figure saved to {save_path}")
        
    plt.close()
    
# 创建四个图表
def create_four_plots(data, save_dir=None):
    # 设置样式
    sns.set_style("whitegrid")
    rcParams['axes.facecolor'] = 'none'
    rcParams['figure.facecolor'] = 'none'
    rcParams['font.size'] = 11  # Set base font size to 11
    
    # 创建 2x2 的图表布局
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()
    
    # 背景圆角矩形
    for ax in axes:
        # 不添加背景矩形，保持透明背景
        pass
    
    # 1. COIN-20 Skill Map (左上)
    primitive_skills = data["primitive_skills"]
    # 按值排序
    sorted_primitive = dict(sorted(primitive_skills.items(), key=lambda x: x[1], reverse=True))
    axes[0].bar(list(sorted_primitive.keys()), list(sorted_primitive.values()), color=COLORS["coin20"], edgecolor=COLORS["edge"], width=0.7)
    axes[0].set_title('Skill Distribution for COIN-20', fontsize=16, pad=20)
    axes[0].tick_params(axis='x', rotation=45, labelsize=11)
    plt.setp(axes[0].get_xticklabels(), ha='right')
    axes[0].tick_params(axis='y', labelsize=11)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # 2. COIN-50 Skill Map (右上)
    interactive_skills = data["interactive_skills"]
    # 按值排序
    sorted_interactive = dict(sorted(interactive_skills.items(), key=lambda x: x[1], reverse=True))
    axes[1].bar(list(sorted_interactive.keys()), list(sorted_interactive.values()), color=COLORS["coin50"], edgecolor=COLORS["edge"], width=0.7)
    axes[1].set_title('Skill Distribution for COIN-50', fontsize=16, pad=20)
    axes[1].tick_params(axis='x', rotation=45, labelsize=11)
    plt.setp(axes[1].get_xticklabels(), ha='right')
    axes[1].tick_params(axis='y', labelsize=11)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # 3. COIN-50 Task Length (左下)
    axes[2].pie(data["task_lengths"], labels=data["task_labels"], colors=[COLORS["coin50"]]*len(data["task_lengths"]),
               autopct='%1.1f%%', startangle=140, textprops={'fontsize': 13},
               wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    axes[2].set_title('Task Length Distribution', fontsize=16, pad=20)
    
    # 4. COIN-50 Subtask Length (右下)
    axes[3].pie(data["subtask_counts"], labels=data["subtask_labels"], colors=[COLORS["coin50"]]*len(data["subtask_counts"]),
               autopct='%1.1f%%', startangle=140, textprops={'fontsize': 13},
               wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    axes[3].set_title('Subtask Count Distribution', fontsize=16, pad=20)
    
    plt.tight_layout(pad=3.0)
    
    # 保存图表
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "coin_benchmark_statistics.png")
        plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.3)
        print(f"Figure saved to {save_path}")
    
    plt.show()

# 主函数
def main():
    # 加载数据
    data = load_data()
    
    # 保存路径
    save_dir = "github_page/static/plots"
    
    # 创建并保存单个图表
    create_single_plot("coin20_skills", data, save_dir)
    create_single_plot("coin50_skills", data, save_dir)
    create_single_plot("task_length", data, save_dir)
    create_single_plot("subtask_count", data, save_dir)
    
    # 创建并保存组合图表
    create_four_plots(data, save_dir)

if __name__ == "__main__":
    main()
