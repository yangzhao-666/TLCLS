import gym
import gym_sokoban

import torch

from common.ActorCritic import ActorCritic

from data_utils import get_agent_data

if __name__ == '__main__':

    data_path = '../maps/1_box/'
    agent_path = '../results/entropy_coef_0.1/1_box/1/pre_train/best_checkpoint'

    ckp = torch.load(agent_path)
    agent = ActorCritic((3,80,80), 5)
    agent.load_state_dict(ckp['a2c'])

    env = gym.make('Curriculum-Sokoban-v2', data_path = data_path)
    get_agent_data(env, 1000, agent)
