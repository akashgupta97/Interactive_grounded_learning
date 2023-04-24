# Interactive_grounded_learning

Reinforcement Learning project by Team Markovian Marvels

## Problem Statement
Build a 3D target structure based on the conversation between an architect (gives instructions) and a builder (asks clarifying questions).

## Installation

Create the python environment
```bash
conda create -y -n rl_project python=3.9
conda activate rl_project
```

All the dependencies can be installed using below command
```bash
pip install -r requirements.txt
```

## Training

To train the NLP T5 Model:
```bash
python nlp_task/train.py
```

To train the RL Model:
```bash
python training_run.py --config_path iglu_baseline.yaml
```

## Evaluation
To evaluate the NLP Model:
```bash
cd nlp_task
python nlp_evaluation.py
```

To evaluate the RL Model:
```bash
python evaluation.py
```