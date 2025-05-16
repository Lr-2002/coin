#!/usr/bin/env python3
"""
Environment Naming System Demo

This script demonstrates how to use the environment naming system to:
1. Parse existing environment names
2. Generate new environment names
3. Filter environments by category
4. Analyze the structure of the benchmark

Usage:
    python env_naming_demo.py
"""

import os
import sys
import argparse
from tabulate import tabulate
from collections import defaultdict

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mani_skill.utils.env_naming import (
    SceneType, ActionType, ObjectType, TaskClass,
    EnvNameParser, EnvNameGenerator, get_all_env_names, get_env_by_category
)


def print_section(title):
    """Print a section title with formatting."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def demo_parse_env_names():
    """Demonstrate parsing environment names."""
    print_section("Parsing Environment Names")
    
    example_envs = [
        "Tabletop-Open-Door-v1",
        "Tabletop-Pick-Objects-InBox-v1",
        "Tabletop-Move-Cube-WithPivot-v1",
        "Tabletop-Stack-Cubes-v0",
        "Tabletop-Put-Cube-IntoMicrowave-v1"
    ]
    
    headers = ["Environment Name", "Scene", "Action", "Object", "Modifier", "Version", "Task Class"]
    rows = []
    
    for env_name in example_envs:
        parsed = EnvNameParser.parse_env_name(env_name)
        task_class = EnvNameParser.get_task_class(env_name)
        
        rows.append([
            env_name,
            parsed.get("scene", ""),
            parsed.get("action", ""),
            parsed.get("object", ""),
            parsed.get("modifier", ""),
            parsed.get("version", ""),
            task_class.value
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def demo_generate_env_names():
    """Demonstrate generating environment names."""
    print_section("Generating Environment Names")
    
    examples = [
        (SceneType.TABLETOP, ActionType.PICK, ObjectType.APPLE, None, 1),
        (SceneType.TABLETOP, ActionType.OPEN, ObjectType.CABINET, "WithSwitch", 1),
        (SceneType.KITCHEN, ActionType.PUT, ObjectType.FORK, "OnPlate", 2),
        (SceneType.OFFICE, ActionType.STACK, ObjectType.CUBES, None, 1),
        (SceneType.WORKSHOP, ActionType.ROTATE, ObjectType.CUBE, "Twice", 1)
    ]
    
    headers = ["Scene", "Action", "Object", "Modifier", "Version", "Generated Name"]
    rows = []
    
    for scene, action, obj, modifier, version in examples:
        generated = EnvNameGenerator.generate_env_name(scene, action, obj, modifier, version)
        rows.append([
            scene.value,
            action.value,
            obj.value,
            modifier or "",
            f"v{version}",
            generated
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def demo_filter_environments():
    """Demonstrate filtering environments by category."""
    print_section("Filtering Environments by Category")
    
    # Get all registered environments
    all_envs = get_all_env_names()
    
    # Filter examples
    filters = [
        ("All Pick Tasks", {"action_type": ActionType.PICK}),
        ("All Door Tasks", {"object_type": ObjectType.DOOR}),
        ("All Primitive Tasks", {"task_class": TaskClass.PRIMITIVE}),
        ("All Interactive Tasks", {"task_class": TaskClass.INTERACTIVE}),
        ("All Tabletop Pick Tasks", {"scene_type": SceneType.TABLETOP, "action_type": ActionType.PICK})
    ]
    
    for filter_name, filter_args in filters:
        filtered_envs = get_env_by_category(**filter_args)
        print(f"\n{filter_name} ({len(filtered_envs)} environments):")
        for env in filtered_envs:
            print(f"  - {env}")


def analyze_benchmark_structure():
    """Analyze the structure of the benchmark."""
    print_section("Benchmark Structure Analysis")
    
    all_envs = get_all_env_names()
    
    # Count by task class
    task_class_counts = defaultdict(int)
    for env in all_envs:
        task_class = EnvNameParser.get_task_class(env)
        task_class_counts[task_class.value] += 1
    
    # Count by action type
    action_counts = defaultdict(int)
    for env in all_envs:
        parsed = EnvNameParser.parse_env_name(env)
        action = parsed.get("action", "Unknown")
        action_counts[action] += 1
    
    # Count by object type
    object_counts = defaultdict(int)
    for env in all_envs:
        parsed = EnvNameParser.parse_env_name(env)
        obj = parsed.get("object", "Unknown")
        object_counts[obj] += 1
    
    # Print task class distribution
    print("\nTask Class Distribution:")
    task_class_rows = [[class_name, count] for class_name, count in task_class_counts.items()]
    print(tabulate(task_class_rows, headers=["Task Class", "Count"], tablefmt="grid"))
    
    # Print action type distribution
    print("\nAction Type Distribution:")
    action_rows = [[action, count] for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True)]
    print(tabulate(action_rows, headers=["Action", "Count"], tablefmt="grid"))
    
    # Print object type distribution
    print("\nObject Type Distribution:")
    object_rows = [[obj, count] for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)]
    print(tabulate(object_rows, headers=["Object", "Count"], tablefmt="grid"))


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="Environment Naming System Demo")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    parser.add_argument("--parse", action="store_true", help="Demo parsing environment names")
    parser.add_argument("--generate", action="store_true", help="Demo generating environment names")
    parser.add_argument("--filter", action="store_true", help="Demo filtering environments")
    parser.add_argument("--analyze", action="store_true", help="Analyze benchmark structure")
    
    args = parser.parse_args()
    
    # If no specific demo is selected, run all
    if not (args.parse or args.generate or args.filter or args.analyze):
        args.all = True
    
    if args.all or args.parse:
        demo_parse_env_names()
    
    if args.all or args.generate:
        demo_generate_env_names()
    
    if args.all or args.filter:
        demo_filter_environments()
    
    if args.all or args.analyze:
        analyze_benchmark_structure()


if __name__ == "__main__":
    main()
