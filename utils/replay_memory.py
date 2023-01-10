import torch

class ReplayMemory:
    def __init__(
        self,
        capacity, num_agent,
        state_dim, action_dim,
        device = 'cpu',
        centralised=False,
        full_sink: bool = True,
    ):
        self.device = device
        self.capacity = capacity
        self.size = 0
        self.pos = 0

        sink = lambda x: x.to(device) if full_sink else x
        if centralised:
            self.states = sink(
                torch.zeros((capacity, state_dim)))
            self.next_states = sink(
                torch.zeros((capacity, state_dim)))
            self.actions = sink(torch.zeros((capacity, action_dim)))
            self.rewards = sink(torch.zeros((capacity,)))
            self.dones = sink(torch.zeros((capacity,), dtype=torch.int8))
        else:
            self.states = sink(
                torch.zeros((capacity, num_agent, state_dim)))
            self.next_states = sink(
                torch.zeros((capacity, num_agent, state_dim)))
            self.actions = sink(torch.zeros((capacity, num_agent, action_dim)))
            self.rewards = sink(torch.zeros((capacity, num_agent)))
            self.dones = sink(torch.zeros((capacity,), dtype=torch.int8))

    def push(self, states, action, reward, next_states, done):
        self.states[self.pos] = states
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward.view(-1)
        self.next_states[self.pos] = next_states
        self.dones[self.pos] = done

        self.pos += 1
        self.size = max(self.size, self.pos)
        self.pos %= self.capacity

    def clear(self):
        self.pos = 0
    
    def sample(self, batch_size):
        indices = torch.randint(0, high=self.size, size=(batch_size,))
        batch_states = self.states[indices].to(self.device)
        batch_actions = self.actions[indices].to(self.device)
        batch_rewards = self.rewards[indices].to(self.device)
        batch_next_states = self.next_states[indices].to(self.device)
        batch_dones = self.dones[indices].to(self.device)
        return (
            batch_states, batch_actions, batch_rewards,
            batch_next_states, batch_dones
        )