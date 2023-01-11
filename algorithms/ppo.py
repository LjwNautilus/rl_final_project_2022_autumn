from abc import abstractclassmethod

import torch
        
class PPO:
    def __init__(
        self,
        num_agents,
        state_dim, hidden_dim, action_dim,
        device='cpu',
        actor_lr=1e-3, 
        critic_lr=1e-3, 
        gae_lambda=0.95, 
        eps=0.2, 
        gamma=0.99,
    ):  
        pass

    # Loads saved weights of networks.
    def load(self, actor_weight_path):
        actor_weight = torch.load(actor_weight_path)
        self.actor.load_state_dict(actor_weight)

    # Computes the action produced by the given network.
    # The network could be the actor network or the target network.
    def network_action(self, network: torch.nn.Module, state, u_range):
        state = state.to(self.device)
        mu, sigma = network(state)
        action_distribution = torch.distributions.Normal(mu, sigma)
        action = action_distribution.sample()
        return torch.clamp(action, -u_range, u_range)

    # Calculates log probabilities of actions.
    def log_probability(self, network: torch.nn.Module, state, action):
        state = state.to(self.device)
        mu, sigma = network(state)
        action_distribution = torch.distributions.Normal(mu, sigma)
        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()
        return log_prob, entropy

    # The api for function run_env(),
    # computes the action produced by the actor network.
    def compute_action(self, state: torch.Tensor, u_range: float):
        return self.network_action(self.actor, state, u_range)

    # Calculates advantages using gae estimation.
    def gae_advantage(self, td_error: torch.Tensor):
        gaes = td_error[:]
        for t in range(len(gaes) - 2, -1, -1):
            gaes[t] += self.gamma * self.gae_lambda * gaes[t + 1]
        return gaes

    # Updates networks.
    @abstractclassmethod
    def update(self, memory, batch_size):
        pass

    @abstractclassmethod
    def learn(
        self, 
        env, 
        epoch=100, 
        sample_num=1024,
        warm_epoch=5,
        policy_update=128,
        update_batch=32,
        evaluate_frequency=10,
        save_frequency=-1,
        write_file=False
    ):
        pass