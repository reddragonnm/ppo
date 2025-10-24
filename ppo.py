import torch
from torch import nn

from actor_critic import ActorCritic

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PPO:
    def __init__(self, lr=3e-4, discount_factor=0.99, clip=0.2):
        self.buffer = self.initialize_buffer()

        self.discount_factor = discount_factor
        self.clip = clip
        self.lr = lr
        self.value_coeff = 0.5
        self.entropy_coeff = 0.1

        self.policy = ActorCritic().to(DEVICE)

        self.policy_old = ActorCritic().to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def initialize_buffer(self):
        return {
            "states": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "is_terminals": [],
            "state_values": [],
        }

    def get_action(self, state):
        state = torch.FloatTensor(state).to(DEVICE).unsqueeze(0)
        action, action_logprob, state_value = self.policy_old(state)

        self.buffer["states"].append(state.detach().cpu())  # prevent GPU memory leak
        self.buffer["actions"].append(action.detach().cpu())
        self.buffer["logprobs"].append(action_logprob.detach().cpu())
        self.buffer["state_values"].append(state_value.detach().cpu())

        return action.item()

    def add_reward(self, reward, done):
        self.buffer["rewards"].append(reward)
        self.buffer["is_terminals"].append(done)

    def update(self, K=4):
        self.policy.train()
        self.policy_old.eval()

        rewards = []
        discounted_reward = 0
        for r, is_terminal in zip(
            reversed(self.buffer["rewards"]), reversed(self.buffer["is_terminals"])
        ):
            discounted_reward = r + (
                self.discount_factor * discounted_reward * int(1 - is_terminal)
            )
            rewards.insert(0, discounted_reward)

        rewards = torch.FloatTensor(rewards).to(DEVICE)

        old_states = torch.cat(self.buffer["states"]).to(DEVICE).squeeze()
        old_actions = torch.cat(self.buffer["actions"]).to(DEVICE).squeeze()
        old_logprobs = torch.cat(self.buffer["logprobs"]).to(DEVICE).squeeze()
        old_state_values = torch.cat(self.buffer["state_values"]).to(DEVICE).squeeze()

        advantages = rewards.detach() - old_state_values.detach()

        for _ in range(K):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            state_values = state_values.squeeze()
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages.detach()
            surr2 = (
                torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages.detach()
            )

            loss_val = (
                -torch.min(surr1, surr2)
                + self.value_coeff * self.loss(state_values, rewards.detach())
                - self.entropy_coeff * dist_entropy
            )

            self.optimizer.zero_grad()
            loss_val.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()

        self.buffer = self.initialize_buffer()
