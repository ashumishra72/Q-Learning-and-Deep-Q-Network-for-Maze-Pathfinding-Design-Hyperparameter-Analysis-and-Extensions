## Overview
This tutorial provides a guide to configuring your Python environment for Q-Learning and Deep Q-Learning. 
We recommend [Anaconda](https://www.anaconda.com/) for its robust package management capabilities.

For installing please refer to [Anaconda download page](https://www.anaconda.com/download), and referring [guide](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda) to started conda.

## Step 1, install Python requirements:

### Create New "RL" Environment:
For Windows: use Anaconda Prompt; for macOS and Linux: use Terminal.
- ``cd xxx/Maze``
- ``conda create -y -n RL python=3.9``
### Activate the "RL" Environment:
- ``conda activate RL``
### Install Dependencies:
- ``pip install -r requirement.txt``

## Step 2, run the experiment:
- ``python main.py``

## Step 3, other options (added for this project):
- ``python main.py QLearning`` or ``python main.py DQN`` chooses the method (default: DQN). After training, one greedy test episode (epsilon = 0) is run; its screenshot is saved as ``screenshot_<method>.png`` and the reward plot as ``reward_<method>.png``.
- ``python run_experiments.py`` runs the Task V hyperparameter study (3 seeds per setting) and writes ``results_summary.csv``, the ``ql_*.png`` / ``dqn_*.png`` plots and ``maze_layout.png``.
- ``python extra_credit.py`` runs the extra-credit experiments EC1-EC6 and writes ``extra_credit_summary.csv`` and the ``ec_*.png`` plots.
- Tested with Python 3.10 (numpy 2.2, torch 2.11 CPU, pygame 2.6.1, matplotlib 3.10). pip found no pygame wheel for Python 3.14 at the time of testing, so use Python 3.9-3.12.
- Without a screen (or to avoid the plot window), set ``SDL_VIDEODRIVER=dummy`` and ``MPLBACKEND=Agg``.