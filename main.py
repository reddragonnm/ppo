import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
import ale_py

import torch

import cv2
import matplotlib.pyplot as plt
from collections import deque

import numpy as np

from ppo import PPO, DEVICE

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", render_mode=None)
env = AtariPreprocessing(
    env, grayscale_obs=True, scale_obs=True, frame_skip=1, terminal_on_life_loss=True
)


def preprocess_obs(obs, crop_top=18):
    obs = cv2.resize(obs, (84, 110))
    return obs[crop_top : crop_top + 84, :]


K = 10
total_frames = 1_000_000_000
max_ep_frames = 20_000
policy_update_freq = 50

ppo = PPO(lr=2e-4)
ppo.policy.load_state_dict(torch.load("models/model_56500.pth", map_location=DEVICE))
ppo.policy_old.load_state_dict(
    torch.load("models/model_56500.pth", map_location=DEVICE)
)
ppo.entropy_coeff = 0.01

frame = 11175972
num_episodes = 56551

max_reward = 51.56

ACTION_MAP = [0, 2, 3]  # NOOP, LEFT, RIGHT

ep_rewards = []

while True:
    env.reset()
    env.step(1)  # FIRE
    obs, _, _, _, _ = env.step(1)  # FIRE

    frame_stack = deque([preprocess_obs(obs)] * 4, maxlen=4)

    ep_reward = 0

    for _ in range(max_ep_frames):
        action_idx = ppo.get_action(np.array(frame_stack))

        action = ACTION_MAP[action_idx]
        obs, reward, terminated, truncated, info = env.step(action)

        reward -= 0.01  # time penalty
        done = terminated or truncated

        if done:
            reward -= 1.0  # penalty for losing the game

        frame_stack.append(preprocess_obs(obs))

        ppo.add_reward(reward, done)

        ep_reward += reward
        frame += 1

        if done:
            break

    if num_episodes % policy_update_freq == 0:
        mean_reward = sum(ep_rewards) / len(ep_rewards)

        print("Training...")
        print(
            f"Buffer size: {len(ppo.buffer['rewards'])} frames Mean reward: {mean_reward:.3f} Max reward: {max(ep_rewards):.3f}"
        )

        # ppo.entropy_coeff = max(0.01, 0.1 - (frame / 1_500_000) * 0.1)
        ppo.update(K=K)

        torch.save(ppo.policy.state_dict(), f"models/model_{num_episodes}.pth")
        print(f"Saved model at episode {num_episodes} frame {frame}")

        ep_rewards = []

    if ep_reward >= max_reward:
        torch.save(ppo.policy.state_dict(), f"models/best_model.pth")
        max_reward = ep_reward

    num_episodes += 1
    ep_rewards.append(ep_reward)

    if num_episodes % 10 == 0:
        print(
            f"Episode {num_episodes} Frame: {frame} Episode Reward: {ep_reward:.3f} Max Reward: {max_reward:.3f} Entropy Coeff: {ppo.entropy_coeff:.3f}"
        )
