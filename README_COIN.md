# COIN Benchmark

## Introduction
Official repository for the COIN Benchmark (Chain of Interaction Benchmark) based on ManiSkill3. COIN Benchmark is designed to evaluate robot manipulation capabilities in both primitive tasks and interactive reasoning scenarios.

The benchmark consists of two main task categories:
1. **Primitive Actions**: Basic manipulation tasks like pick, put, open, close, lift, rotate, and stack. Contains 20 tasks.
2. **Interactive Reasoning**: More complex tasks requiring decision-making and understanding of the environment. Contains 50 tasks.

## Models

The benchmark supports multiple Vision-Language-Action (VLA) models:

### Pi0 
- Repository: [https://github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- A foundation model for robotic manipulation by Physical Intelligence

### CogACT 
- Repository: [https://github.com/microsoft/CogACT](https://github.com/microsoft/CogACT)
- A foundational model for synergizing cognition and action in robotic manipulation by Microsoft

### Gr00t 
- Repository: [https://github.com/NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- NVIDIA's foundation model for generalized humanoid robot reasoning and skills.

### Resource Requirements

| Model  | Disk Space | GPU Memory for Inference |
|--------|------------|--------------------------|
| Pi0    | 41GB       | > 8 GB                   |
| CogACT | 30GB       | > 25 GB                  |
| Gr00t  | 18GB       | > 5 GB                   |

We run all the evaluation on NVIDIA A800.
## Preparation

Install the COINBench repository
```bash
git clone --recurse-submodules https://github.com/Lr-2002/COINBench_NeurIPS2025.git

# if you have already cloned the repository, run the following command to update the submodules
cd COINBench_NeurIPS2025
git submodule update --init --recursive
```

Install ManiSkill3
```bash
# install the package
pip install --upgrade mani_skill
# install a version of torch that is compatible with your system
pip install torch
```

You also need to set up [Vulkan](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan) with instructions here


## Teleoperation for data collection

The benchmark supports mujoco teleoperation on different devices:

1. **MujocoAR**: [https://github.com/omarrayyann/MujocoAR](https://github.com/omarrayyann/MujocoAR)

2. **ARCore Android SDK**: [https://github.com/Lr-2002/arcore-android-sdk/tree/main?tab=readme-ov-file](https://github.com/Lr-2002/arcore-android-sdk/tree/main?tab=readme-ov-file)

3. **Python Teleoperation**:
   ```bash
    python -m mani_skill.examples.teleoperation.mujoco_ar_teleop -e Tabletop-PickPlace-Apple-v1

   ```
   Then open your mujoco ar app, set the address and port to the one in the terminal, and start the teleoperation.

## Inference

Evaluation requires running two terminals - one for the VLA server and one for the ManiSkill client.

### VLA Inference

#### Pi0

**Pi0 server**:
```bash
uv run scripts/serve_policy.py --port=8000 policy:checkpoint --policy.config=pi0_fast_coin_primitive --policy.dir=checkpoints/pi0_fast_coin_primitive/pi0_fast_coin_primitive_v4/470000/
```

**Maniskill client**:
```bash
xvfb-run -a python env_tests/run_all_tasks_vla.py --vla-agent pi0 --num-episodes 10 --max-steps 400 --port 8000 --cameras human_camera base_front_camera hand_camera --output-dir evaluation/pi0_primitive --primitive
```

#### Gr00t

**Gr00t server**:
```bash
python test_connect.py --model_path "checkpoints/primitive_dataset_v4/checkpoint-120000/" --port 8001
```

**Maniskill client**:
```bash
xvfb-run -a python env_tests/run_all_tasks_vla.py --vla-agent gr00t --num-episodes 10 --max-steps 400 --port 8001 --cameras human_camera base_front_camera hand_camera --output-dir evaluation/gr00t_primitive --primitive
```

#### CogACT

**CogACT server**:
```bash
python test_connect.py
```

**Maniskill client**:
```bash
xvfb-run -a python env_tests/run_all_tasks_vla.py --vla-agent cogact --num-episodes 10 --max-steps 400 --port 8002 --cameras human_camera --output-dir evaluation/cogact_primitive --primitive
```

### Hierarchical VLA

**Pi0**:
```bash
xvfb-run -a python env_tests/run_all_tasks_vla.py --vla-agent pi0 --num-episodes 10 --max-steps 400 --port 8003 --cameras human_camera base_front_camera hand_camera --output-dir evaluation/pi0_interactive_gpt-4o --interactive --hierarchical --llm-model gpt-4o
```

### Code-As-Policy
The benchmark also supports Code-As-Policy approaches for evaluation.

## Results

### Processing Results
Use the following tools to process and analyze evaluation results:

To measure the success rate of the models, first you need to change the `input_path`(for example, `evaluation/pi0_primitive`) in `tools/measure_evaluation.py` to the path of the evaluation results, then run:

```
python tools/measure_evaluation.py 
```

The output csv file will be saved in the `input_path` you specified.


## Environment Table

The COIN benchmark includes a diverse set of environments:

| Name | Category | Description |
|------|----------|-------------|
| Tabletop-Close-Cabinet-v1 | Primitive | close the cabinet door |
| Tabletop-Open-Microwave-v1 | Primitive | open the microwave |
| Tabletop-Pick-Pen-v1 | Primitive | pick up the pen and put it to the marker |
| Tabletop-Put-Fork-OnPlate-v1 | Primitive | put the fork on the plate |
| Tabletop-Stack-Cubes-v1 | Primitive | stack all the cube |
| Tabletop-Find-Seal-v1 | Interactive | Find the cube which have red face downward, and put it on the marker with red face upward |
| Tabletop-Close-Door-WithObstacle-v1 | Interactive | close the door |
| Tabletop-Balance-Pivot-WithBalls-v1 | Interactive | Put the balls in to the holder to balance the long board on the triangular prism |
| *and many more* | | |

For a complete list of environments, refer to the [environment directory](mani_skill/envs/tasks/coin_bench/).

## License

All the environments in COINBench are licensed under the Apache License 2.0.
