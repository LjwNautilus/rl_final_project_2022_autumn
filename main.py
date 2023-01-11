import sys
import torch
import vmas
from vmas.scenarios.balance import HeuristicPolicy as BalanceHeuristic
from vmas.scenarios.transport import HeuristicPolicy as TransportHeuristic
from vmas.scenarios.wheel import HeuristicPolicy as WheelHeuristic

from ippo import IPPO
from cppo import CPPO
from utils.run_env import run_env

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Environment hyperparameters
MAX_STEPS = None
NUM_ENVS = 1
RENDER = True
# Scenario name
SCENARIO = 'balance'
# Network hyperparameters
HIDDEN_DIM = 8
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
# Learning hyperparameters
EPOCH = 400
SAMPLE_NUM = 256
WARM_EPOCH = 10
POLICY_UPDATE = 64
UPDATE_BATCH = 32
EVALUATE_FREQUENCY = 10
SAVE_FREQUENCY = -1
WRITE_FILE = True

ALGORITHM = 'ippo'

# Fixs random seed.
torch.manual_seed(0)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        SCENARIO = sys.argv[1]
    if len(sys.argv) > 2:
        temp = sys.argv[2].lower()
        if temp == 'cppo':
            ALGORITHM = 'cppo'
    if len(sys.argv) > 3:
        DEVICE = sys.argv[3]
    
    env = vmas.make_env(
        scenario_name=SCENARIO,
        num_envs=NUM_ENVS,
        device=DEVICE,
        continuous_actions=True,
        wrapper=None,
        max_steps=MAX_STEPS,
    )
    obs = env.reset()
    state_dim = obs[0][0].shape[0]
    batch_size = env.batch_dim
    action_dim = env.get_agent_action_size(env.agents[0])
    num_agents = env.n_agents
    
    if ALGORITHM == 'ippo':
        agent = IPPO(
            num_agents, 
            state_dim, HIDDEN_DIM, action_dim,
            actor_lr=ACTOR_LR, 
            critic_lr=CRITIC_LR,
            device=DEVICE
        )
    else:
        agent = CPPO(
            num_agents, 
            state_dim, HIDDEN_DIM, action_dim,
            actor_lr=ACTOR_LR, 
            critic_lr=CRITIC_LR,
            device=DEVICE
        )

    agent.learn(
        env, 
        epoch=EPOCH, 
        sample_num=SAMPLE_NUM, 
        warm_epoch=WARM_EPOCH,
        policy_update=POLICY_UPDATE,
        update_batch=UPDATE_BATCH,
        evaluate_frequency=EVALUATE_FREQUENCY,
        save_frequency=SAVE_FREQUENCY,
        write_file=WRITE_FILE
    )

    if ALGORITHM == 'cppo':
        reward = run_env(
            env, agent, n_steps=1000, render=RENDER, centralised=True
        )
    else:
        reward = run_env(env, agent, n_steps=1000, render=RENDER)
    print(f'Final {ALGORITHM} reward: ', reward)

    if SCENARIO == 'wheel':
        heuristic = WheelHeuristic(continuous_action=True)
    elif SCENARIO == 'balance':
        heuristic = BalanceHeuristic(continuous_action=True)
    else:
        heuristic = TransportHeuristic(True)
    heuristic_reward = run_env(env, heuristic, render=False)
    print('Heuristic reward: ', heuristic_reward)