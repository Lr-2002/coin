import os
import json
import csv
from datetime import datetime


def process_metadata_folder(root_path):
    results = []

    # 遍历根目录下的所有文件夹
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)

        # 检查是否是文件夹且包含metadata.json
        metadata_path = os.path.join(folder_path, "metadata/metadata.json")
        if os.path.isdir(folder_path) and os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                    # 提取所需信息
                    model_class = metadata.get(
                        "model", {}).get("class", "unknown")
                    env_name = metadata.get(
                        "environment", {}).get("name", "unknown")
                    timestamp = metadata.get("timestamp", "unknown")
                    success = metadata.get("success", False)

                    # 解析时间戳为更易读的格式
                    try:
                        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        formatted_time = dt.strftime("%Y%m%d_%H%M%S")
                    except:
                        formatted_time = timestamp

                    results.append(
                        {
                            "model": model_class,
                            "environment": env_name,
                            "timestamp": formatted_time,
                            "success": success,
                        }
                    )
            except Exception as e:
                print(f"Error processing {metadata_path}: {str(e)}")

    return results


def calculate_success_rate(results):
    # 按环境任务分组统计成功率
    task_stats = {}

    for result in results:
        env = result["environment"]
        if env not in task_stats:
            task_stats[env] = {"total": 0, "success": 0}

        task_stats[env]["total"] += 1
        if result["success"]:
            task_stats[env]["success"] += 1

    # 计算成功率
    success_rates = []
    for env, stats in task_stats.items():
        rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        success_rates.append(
            {
                "task": env,
                "success_rate": f"{rate:.2%}",
                "success_count": stats["success"],
                "total_count": stats["total"],
            }
        )

    return success_rates


def save_results_to_csv(results, success_rates, output_path, duplicate_path=None):
    # 获取当前时间作为文件名的一部分
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = results[0]["model"] if results else "unknown"
    filename = f"{model_name}_{current_time}.csv"
    filepath = os.path.join(output_path, filename)

    # 写入CSV文件
    with open(filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # 写入标题
        writer.writerow(
            ["Task", "Success Rate", "Success Count", "Total Count"])

        # 写入数据
        for rate in success_rates:
            writer.writerow(
                [
                    rate["task"],
                    rate["success_rate"],
                    rate["success_count"],
                    rate["total_count"],
                ]
            )

    print(f"Results saved to {filepath}")

    if duplicate_path:
        import shutil

        os.makedirs(duplicate_path, exist_ok=True)
        shutil.copy(filepath, duplicate_path)

        # Create subfolder in the duplicated path and copy origin source data and csv to this folder

        # Copy the duplicated CSV to the subfolder
        shutil.copytree(
            os.path.dirname(filepath), os.path.join(
                duplicate_path, "origin_data")
        )
        # TODO: Specify the path to the original source data file
        # Example: source_data_path = '/path/to/source_data.file'
        # shutil.copy(source_data_path, os.path.join(subfolder, os.path.basename(source_data_path)))


def main():
    # 输入路径
    # input_path = input("请输入包含metadata文件夹的路径: ")

    # input_path = "/home/lr-2002/project/reasoning_manipulation/Manigen/evaluation/2025-05-08_23-08-55/"
    # input_path = "/home/lr-2002/project/reasoning_manipulation/ReKep/evaluation/2025-05-05_20-07-31/"
    # input_path = "evaluation/20250507_pi0_350000/"
    # input_path = "evaluation/20250507_cogact_40000/"
    # input_path = "evaluation/20250509_pi0_470000"

    # input_path = "evaluation/20250515_cogact_30000"
    # input_path = "evaluation/20250515_gr00t_120000"
    # input_path = "evaluation/20250515_pi0_470000"

    # input_path = "evaluation/20250515_gr00t_120000_close_cabinet"
    # input_path = "evaluation/20250515_cogact_30000_close_cabinet"
    # input_path = "evaluation/20250515_pi0_470000_close_cabinet"
    
    # input_path = "evaluation/20250516_cogact_30000_open_cabinet"
    # input_path = "evaluation/20250516_gr00t_120000_open_cabinet"
    # input_path = "evaluation/20250516_pi0_470000_open_cabinet"
    
    # input_path = "evaluation/20250516_gr00t_03pri"
    # input_path = "evaluation/20250516_cogact_03pri"
    input_path = "evaluation/20250516_pi0_03pri"
    results = process_metadata_folder(input_path)

    if not results:
        print("没有找到任何有效的metadata文件")
        return

        # 计算成功率
    success_rates = calculate_success_rate(results)
    time_stamp = results[0]["timestamp"]
    # 保存结果
    save_results_to_csv(
        results, success_rates, input_path, "./evaluation_results/" + time_stamp + "/"
    )


if __name__ == "__main__":
    main()
