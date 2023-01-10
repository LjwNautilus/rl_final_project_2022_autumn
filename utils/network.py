import torch

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.fc_mu = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Tanh()
        )
        self.fc_sigma = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Softplus()
        )

    def forward(self, x):
        h = self.fc2(self.fc1(x))
        mu = self.fc_mu(h)
        sigma = self.fc_sigma(h)
        return mu, sigma

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Flatten(0)
        )

    def forward(self, x):
        return self.out(self.fc2(self.fc1(x)))