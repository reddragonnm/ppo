import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing
import torch
from collections import deque
import numpy as np
import cv2

from torch import nn
from torch.distributions.categorical import Categorical


from actor_critic import ActorCritic
from ppo import PPO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Preprocess observation
def preprocess_obs(obs, crop_top=18):
    obs = cv2.resize(obs, (84, 110))
    return obs[crop_top : crop_top + 84, :]


ACTION_MAP = [0, 2, 3]  # NOOP, LEFT, RIGHT

gym.register_envs(ale_py)

# Load environment
env = gym.make("ALE/Breakout-v5", render_mode="human")
env = AtariPreprocessing(
    env, grayscale_obs=True, scale_obs=True, frame_skip=1, terminal_on_life_loss=True
)

# Load trained policy
policy = ActorCritic().to(DEVICE)
policy.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
policy.eval()

done = False
obs, info = env.reset()
obs, _, _, _, _ = env.step(1)  # FIRE
obs, _, _, _, _ = env.step(1)  # FIRE

frame_stack = deque([preprocess_obs(obs)] * 4, maxlen=4)

total_reward = 0

while True:
    # Stack frames
    state = np.stack(frame_stack, axis=0)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        action, _, _ = policy(state_tensor)

    action_env = ACTION_MAP[action.item()]
    obs, reward, terminated, truncated, info = env.step(action_env)
    done = terminated or truncated

    reward -= 0.01  # time penalty
    if done:
        reward -= 1.0  # penalty for losing the game

    frame_stack.append(preprocess_obs(obs))
    total_reward += reward

    if done:
        print("Episode reward:", total_reward)
        total_reward = 0
        obs, info = env.reset()
        obs, _, _, _, _ = env.step(1)
        obs, _, _, _, _ = env.step(1)
        frame_stack = deque([preprocess_obs(obs)] * 4, maxlen=4)
