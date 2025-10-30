# Atari Breakout PPO Agent

This repository implements a Proximal Policy Optimization (PPO) agent to play Atari Breakout using PyTorch and Gymnasium. It includes training, evaluation, and video rendering utilities.

![Breakout Gameplay](videos/best_run.mp4)

## Features

- **PPO Algorithm**: Stable policy gradient method ([`ppo.py`](ppo.py)).
- **Actor-Critic Model**: CNN-based policy/value network ([`actor_critic.py`](actor_critic.py)).
- **Frame Preprocessing**: Cropping, resizing, grayscale conversion.
- **Training Script**: Train agent and save checkpoints ([`main.py`](main.py), [`main.ipynb`](main.ipynb)).
- **Evaluation Script**: Run trained agent interactively ([`test.py`](test.py)).
- **Video Rendering**: Generate crisp, squashed, slow-motion videos ([`make_best_video.py`](make_best_video.py)).
- **Best Model**: Pretrained weights at [`models/best_model.pth`](models/best_model.pth).
- **Best Run Video**: Example gameplay at [`videos/best_video.mp4`](videos/best_video.mp4).

## Getting Started

### Requirements

- Python 3.10+
- PyTorch
- Gymnasium
- ALE-py
- OpenCV
- imageio

Install dependencies:

```sh
pip install torch gymnasium[atari] ale-py opencv-python imageio
```

### Training

To train the agent:

```sh
python main.py
```

or use [`main.ipynb`](main.ipynb) for interactive training.

Checkpoints are saved in [`models/`](models/).

### Evaluation

To run the trained agent:

```sh
python test.py
```

This loads [`models/best_model.pth`](models/best_model.pth) and plays Breakout.

### Video Generation

To render a video of the best run:

```sh
python make_best_video.py
```

Output is saved to [`videos/`](videos/) (see [`videos/best_video.mp4`](videos/best_video.mp4)).

## File Overview

- [`ppo.py`](ppo.py): PPO algorithm implementation ([`PPO`](ppo.py)).
- [`actor_critic.py`](actor_critic.py): CNN actor-critic model ([`ActorCritic`](actor_critic.py)).
- [`main.py`](main.py): Training loop.
- [`main.ipynb`](main.ipynb): Notebook for training/experimentation.
- [`test.py`](test.py): Evaluation script.
- [`make_best_video.py`](make_best_video.py): Video rendering utility.
- [`models/`](models/): Saved model checkpoints.
- [`videos/`](videos/): Rendered gameplay videos.

## Example Results

- **Best Model**: [`models/best_model.pth`](models/best_model.pth)
- **Best Video**: [`videos/best_video.mp4`](videos/best_video.mp4)

## License

MIT License

---

**References**:

- [OpenAI PPO Paper](https://arxiv.org/abs/1707.06347)
- [Gymnasium](https://gymnasium.farama.org/)
- [ALE-py](https://github.com/Farama-Foundation/ale-py)
