import gym
import gym_sokoban

from common.fix_and_reinit import fix_and_reinit
from common.train_the_agent import train_the_agent
from common.ActorCritic import ActorCritic
from common.RolloutStorage import RolloutStorage
from common.multiprocessing_env import SubprocVecEnv

import torch
import torch.autograd as autograd
import torch.optim as optim

import argparse
import wandb

def train(args, wandb_session):

    if args.task == 's1t1fc_game2':
        fix = 'conv'
        source_env_name = 'Curriculum-Sokoban-v2'
        taget_env_name = 'Curriculum-Sokoban-v4'
    else:
        fix = args.task[5]
        source_env_name = 'Curriculum-Sokoban-v2'
        target_env_name = 'Curriculum-Sokoban-v2'

    source_task = args.task[1]
    target_task = args.task[3]
    if source_task == target_task:
        target_task = str(target_task) + '_2'

    source_task_map = args.map_file + str(source_task)
    target_task_map = args.map_file + str(target_task)

    #source task training
    def make_env():
        def _thunk():
            env = gym.make(source_env_name, data_path = source_task_map)
            return env
        return _thunk

    envs = [make_env() for i in range(args.num_envs)]
    envs = SubprocVecEnv(envs)
    state_shape = (3, 80, 80)

    num_actions = 5
    actor_critic = ActorCritic(state_shape, num_actions=num_actions)
    rollout = RolloutStorage(args.rolloutStorage_size, args.num_envs, state_shape)
    optimizer = optim.RMSprop(actor_critic.parameters(), lr=args.lr, eps=args.eps, alpha=args.alpha)

    if args.USE_CUDA:
        if not torch.cuda.is_available():
            raise ValueError('You wanna use cuda, but the machine you are on doesnt support')
        elif torch.cuda.is_available():
            Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda()
            actor_critic.cuda()
            rollout.cuda()
    else:
        Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)


    print('Pre-training the agent...')
    train_the_agent(envs, args.num_envs, Variable, state_shape, actor_critic, optimizer, rollout, data_path=None, args=args, wandb_session=wandb_session) #train and save the model;

    #target task training
    def make_env():
        def _thunk():
            env = gym.make(target_env_name, data_path = target_task_map)
            return env
        return _thunk

    envs = [make_env() for i in range(args.num_envs)]
    envs = SubprocVecEnv(envs)

    actor_critic, optimizer = fix_and_reinit(actor_critic, optimizer, fix)
    print('Fine-tunning the agent...')
    train_the_agent(envs, args.num_envs, Variable, state_shape, actor_critic, optimizer, rollout, data_path=target_task_map, args=args, wandb_session=wandb_session) #train and save the model;

if __name__ == "__main__":
    description = 'TLCLS'
    parser = argparse.ArgumentParser(description=description)
    #parser.add_argument('--num_steps', type=int, default=1000000)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--task', type=str, default='s1t1k1')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy_coef', type=float, default=0.1)
    parser.add_argument('--value_loss_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--rolloutStorage_size', type=int, default=5)
    parser.add_argument('--num_envs', type=int, default=30)
    parser.add_argument('--eval_freq', type=int, default=1000)
    parser.add_argument('--eval_num', type=int, default=20)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--map_file', type=str, default='./maps/')

    args = parser.parse_args()

    args.USE_CUDA = True

    for run in range(args.runs):
        wandb_session = wandb.init(project='TLCLS', config=vars(args), name="run-%i"%(run), reinit=True, group=args.task, mode='disabled')

        config = wandb.config
        train(args, wandb_session)
        wandb_session.finish()
