import argparse

import torch
import vmas
from vmas.scenarios.balance import HeuristicPolicy as BalanceHeuristic
from vmas.scenarios.transport import HeuristicPolicy as TransportHeuristic
from vmas.scenarios.wheel import HeuristicPolicy as WheelHeuristic

from algorithms.cppo import CPPO
from algorithms.ippo import IPPO
from algorithms.mappo import MAPPO
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

ALGORITHM = 'cppo'

# Fixs random seed.
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--scenario',
        dest='scenario',
        help=f'''name of the scenario, default value is \'{SCENARIO}\'.
        valid inputs: [balance | wheel | transport]''',
        default=SCENARIO
    )
    parser.add_argument(
        '-a', '--alg', 
        dest='alg',
        help=f'''name of the algorithm, default value is \'{ALGORITHM}\'.
        valid inputs: [cppo | ippo | mappo]''',
        default=ALGORITHM
    )
    parser.add_argument(
        '-d', '--device',
        dest='device',
        help='the device to be run, valid inputs: [cuda | cuda:$ID | cpu]',
        default=DEVICE
    )
    parser.add_argument(
        '-w', '--write-file',
        dest='write',
        help=f'whether writes rewards to files, default value is {WRITE_FILE}',
        default=WRITE_FILE, type=bool
    )
    parser.add_argument(
        '--save',
        dest='save',
        help=f'''the frequency of saving weights, 
        default value is {SAVE_FREQUENCY}.
        negative number indicates not to save weights.''',
        default=SAVE_FREQUENCY, type=int
    )
    parser.add_argument(
        '-r', '--render',
        dest='render', action='store_true',
        help=f'whether render gif, GL is needed. default is Fasle',
        default=False
    )
    parser.add_argument(
        '-e', '--epoch',
        dest='epoch',
        help=f'number of epoches to be run, default is {EPOCH}',
        default=EPOCH, type=int
    )
    return parser

if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    ALGORITHM = args.alg
    if args.epoch > 0:
        EPOCH = args.epoch
    DEVICE = args.device
    RENDER = args.render
    SCENARIO = args.scenario
    if args.save != 0:
        SAVE_FREQUENCY = args.save
    WRITE_FILE = args.write

    print(f'Algorithm: {ALGORITHM}')
    print(f'Scenario: {SCENARIO}')

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
    elif ALGORITHM == 'cppo':
        agent = CPPO(
            num_agents, 
            state_dim, HIDDEN_DIM, action_dim,
            actor_lr=ACTOR_LR, 
            critic_lr=CRITIC_LR,
            device=DEVICE
        )
    else:
        agent = MAPPO(
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