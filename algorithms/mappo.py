from copy import deepcopy
import os
import sys
from tqdm import tqdm

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__) + '/..')
from ppo import PPO
from utils.network import PolicyNet, ValueNet
from utils.replay_memory import ReplayMemory
from utils.run_env import run_env
        
class MAPPO(PPO):
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
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps = eps
        self.device = device

        self.actors = tuple(
            PolicyNet(state_dim, hidden_dim, action_dim).to(device)
            for _ in range(num_agents)
        )
        self.targets = tuple(
            PolicyNet(state_dim, hidden_dim, action_dim).to(device)
            for _ in range(num_agents)
        )
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizers = tuple(
            torch.optim.AdamW(self.actors[i].parameters(), lr=actor_lr)
            for i in range(num_agents)
        )
        self.critic_optimizer = \
            torch.optim.AdamW(self.critic.parameters(), lr=critic_lr)

    # Loads saved weights of networks.
    def load(self, actor_weight_paths: list):
        actor_weights = [
            torch.load(actor_weight_paths[i])
            for i in range(len(actor_weight_paths))
        ]
        for i in range(len(actor_weights)):
            self.actors[i].load_state_dict(actor_weights[i])

    # The api for function run_env(),
    # computes the action produced by the actor network.
    def compute_action(self, states: list, u_range: float) -> list:
        actions = [None] * len(states)
        for i in range(len(states)):
            actions[i] = self.network_action(self.actors[i], states[i], u_range)
        return actions

    # Updates networks.
    def update(self, memory: ReplayMemory, batch_size):
        actor_loss_list = []
        critic_loss_list = []
        loss_list = []
        states, actions, rewards, next_states, dones = memory.sample(batch_size)
        # Each agent is trained independently but shares the same policy.
        for agent_id in range(self.num_agents):
            agent_states = states[:, agent_id, :]
            agent_next_states = next_states[:, agent_id, :]
            agent_rewards = rewards[:, agent_id]
            agent_actions = actions[:, agent_id, :]
            # Calculates td_target, td_delta and gae advantage.

            value_next = self.critic(agent_next_states)
            value_estimate = self.critic(agent_states)
            td_target = self.gamma * value_next * (1 - dones) + agent_rewards
            td_error = td_target - value_estimate
            advantage = self.gae_advantage(td_error.cpu()).to(self.device)

            old_log_probs, _ = self.log_probability(
                self.targets[agent_id], agent_states, agent_actions
            )
            log_probs, entropy = self.log_probability(
                self.actors[agent_id], agent_states, agent_actions
            )
            prob_ratio = torch.exp(log_probs - old_log_probs)
            # Calculates policy surrogate.
            surrogate = prob_ratio * advantage.view(-1, 1)
            surrogate_clipped = torch.clamp(
                prob_ratio, 1 - self.eps, 1 + self.eps
            ) * advantage.view(-1, 1)
            # Calculates loss using clipped PPO method.
            actor_loss = -torch.min(surrogate, surrogate_clipped).mean()
            critic_loss = F.mse_loss(value_estimate, td_target).mean()
            loss = actor_loss + 0.5 * critic_loss + 0.001 * entropy.mean()
            # Updates actor and critic network.
            self.actor_optimizers[agent_id].zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actors[agent_id].parameters(), 0.5
            )
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.actor_optimizers[agent_id].step()
            self.critic_optimizer.step()
            loss_list.append(loss.item())
            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())
        return loss_list, actor_loss_list, critic_loss_list

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
        memory = ReplayMemory(
            sample_num * warm_epoch, self.num_agents, 
            self.state_dim, self.action_dim, self.device
        )
        evaluate_env = deepcopy(env)
        update_counter = 1
        save_weight = save_frequency > 0
        if save_weight:
            save_prefix = './models'
            if not os.path.exists(save_prefix):
                os.mkdir(save_prefix)
        for episode in tqdm(range(epoch)):
            obs = env.reset()
            episode_reward = 0
            training = episode > warm_epoch
            for step in range(sample_num):
                actions = [None] * len(obs)
                for id in range(len(obs)):
                    actions[id] = self.network_action(
                        self.targets[id],
                        obs[id],
                        u_range=env.agents[id].u_range
                    )
                next_obs, rews, dones, _ = env.step(actions)
                actions = torch.stack(actions, dim=1)
                rewards = torch.stack(rews, dim=1)
                states = torch.stack(obs, dim=1)
                next_states = torch.stack(next_obs, dim=1)
                memory.push(states, actions, rewards, next_states, dones)
                episode_reward += rewards.mean().item()
                if (((step + 1) % policy_update == 0 or step == sample_num - 1)
                    and training):
                    self.update(memory, update_batch)
            # Copies parameters from actor network to target policy network.
            for i in range(self.num_agents):
                self.targets[i].load_state_dict(self.actors[i].state_dict())
            # Writes reward values to file or just prints values.
            if write_file:
                write_str = (f'{update_counter:6d} {episode_reward:.3f}\n')
                with open('train_reward.txt', 'a') as file:
                    file.write(write_str)
                update_counter += 1
            else:
                print('\nTrain reward: ', episode_reward)
            if episode % evaluate_frequency == 0:
                evaluate_reward = run_env(
                    evaluate_env, self, 
                    render=False,
                    alg='mappo'
                )
                if write_file:
                    write_str = (f'{episode // evaluate_frequency:4d}'
                                 + f' {episode:6d}'
                                 + f' { evaluate_reward:.3f}\n')
                    with open('evaluate_reward.txt', 'a') as file:
                        file.write(write_str)
                else:
                    print('\n' + '*' * 36)
                    print('Evaluation reward: ', evaluate_reward)
                    print('*' * 36)
            # Saves weights of networks.
            if save_weight and \
               ((episode + 1) % save_frequency == 0 or episode == epoch - 1):
                critic_weight_name = f'{save_prefix}/critic{episode}.pt'
                torch.save(self.critic.state_dict(), critic_weight_name)
                for id in range(self.num_agents):
                    actor_weight_name = f'{save_prefix}/actor{id}{episode}.pt'
                    torch.save(self.actors[id].state_dict(), actor_weight_name)