# 🌈 Rainbow Learning for Atari Games

![Rainbow Algorithm](https://img.shields.io/badge/Algorithm-Rainbow-ff69b4)
![Game](https://img.shields.io/badge/Game-Ms_Pac--Man-yellow)
![Python Version](https://img.shields.io/badge/Python-3.7+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

<p align="center">
  <img src="https://raw.githubusercontent.com/google-deepmind/bsuite/master/bsuite/experiments/atari_rainbow/rainbow_plots/rainbow.gif" width="400" alt="Rainbow Agent Playing Atari">
</p>

## 📋 Overview

This project implements the Rainbow algorithm for deep reinforcement learning in Atari games, specifically Ms. Pac-Man. Rainbow combines several improvements to Deep Q-Learning:

- ✨ **Dueling Network Architecture**
- 🔢 **Distributional RL (C51)**
- 🎯 **Prioritized Experience Replay**
- 🔄 **Double Q-Learning**
- 📊 **Multi-step Learning**

## 🚀 Getting Started

### Prerequisites

```bash
pip install gymnasium
pip install ale-py
pip install torch
pip install matplotlib
pip install pillow
```

### Training an Agent

```bash
python atari.py
```

This will start training the Rainbow agent on Ms. Pac-Man. The agent's progress will be displayed, and checkpoints will be saved every 100 episodes.

### Watching a Trained Agent Play

```bash
python atariplayback.py
```

This will load a saved model checkpoint and play Ms. Pac-Man, showing the agent's performance visually.

## 🧠 Architecture

The implementation features:

- **DuelingQNetwork**: Separates value and advantage streams for better performance
- **PrioritizedReplayBuffer**: Samples important transitions more frequently
- **Distributional Learning**: Models a distribution over returns instead of expected returns

## 📈 Performance

Training metrics are displayed during training and can be visualized with matplotlib:

- Episode scores
- Running average score
- Training time

## 🔧 Configuration

Key hyperparameters can be adjusted in `atari.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| BUFFER_SIZE | 100000 | Size of replay buffer |
| BATCH_SIZE | 64 | Minibatch size |
| GAMMA | 0.99 | Discount factor |
| LR | 0.0005 | Learning rate |
| N_ATOM | 51 | Number of atoms for distributional RL |

## 📄 License

This project is available under the MIT License.

## 🙏 Acknowledgments

- Deep learning framework: [PyTorch](https://pytorch.org/)
- Reinforcement learning environments: [Gymnasium](https://gymnasium.farama.org/)
- Atari environments: [ALE-Py](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
- Rainbow algorithm: [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)

---

<p align="center">
  <i>Last updated: March 21, 2025</i>
</p>