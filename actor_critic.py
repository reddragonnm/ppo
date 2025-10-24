import torch
from torch import nn
from torch.distributions.categorical import Categorical


class ActorCritic(nn.Module):
    def __init__(self, output_dim=3):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
        )  # from NatureCNN

        self.actor = nn.Linear(512, output_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.feature_extractor(x)

        action_logits = self.actor(x)
        dist = Categorical(logits=action_logits)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action, action_logprob, self.critic(x)

    def evaluate(self, x, action):
        x = self.feature_extractor(x)

        action_logits = self.actor(x)
        dist = Categorical(logits=action_logits)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, self.critic(x), dist_entropy
