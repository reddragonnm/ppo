import torch
from torch import nn
from torch.distributions.categorical import Categorical


class ActorCritic(nn.Module):
    def __init__(self, output_dim=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),  # 84x84 - 20x20
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # 20x20 - 9x9
            nn.ReLU(),
            nn.Flatten(),
        )  # sharing common encoder between actor and critic

        self.actor = self.get_dense_network(out_dim=output_dim)  # remove FIRE
        self.critic = self.get_dense_network(out_dim=1)

    def get_dense_network(self, out_dim):
        return nn.Sequential(
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        x = self.conv(x)

        action_logits = self.actor(x)
        dist = Categorical(logits=action_logits)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action, action_logprob, self.critic(x)

    def evaluate(self, x, action):
        x = self.conv(x)

        action_logits = self.actor(x)
        dist = Categorical(logits=action_logits)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, self.critic(x), dist_entropy
