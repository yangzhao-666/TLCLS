#This code is from openai baseline
#https://github.com/openai/baselines/tree/master/baselines/common/vec_env

import numpy as np
from multiprocessing import Process, Pipe
import gym
import gym_sokoban

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            map_file = None
            if done:
                ob = env.reset(data)
                map_file = env.map_file
                if env.num_env_steps < env.max_steps:
                    win_lose = 1
                else:
                    win_lose = 0
            remote.send((ob, reward, done, info, win_lose, map_file))
        elif cmd == 'reset':
            ob = env.reset(data)
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    
class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, data_path, gamma=0.5, alpha=0.5, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        #implementation of difficulty quantum momentum;
        self.gamma = gamma
        self.alpha = alpha
        self.dqn_dict, self.N = initialize_dqn_dict(data_path, self.gamma)
        self.maps = self.dqn_dict.keys()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, win_loses, map_files = zip(*results)
        
        for win_lose, map_file in zip(win_loses, map_files):
            if map_file is not None:
                h = self.dqn_dict[map_file][0]
                r = self.dqn_dict[map_file][1]
                w = self.dqn_dict[map_file][2]
                p = self.dqn_dict[map_file][3]

                h = self.alpha * h + (1-self.alpha) * win_lose
                r = (1 - h) ^2
                
                theta = r / p
                w = w ^(self.gamma * theta / self.N)

                p = (1 - self.gamma) + self.gamma/self.N

                self.dqn_dict[map_file][0] = h
                self.dqn_dict[map_file][1] = r
                self.dqn_dict[map_file][2] = w
                self.dqn_dict[map_file][3] = p

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        #when reset, select the most suitable map for the worker
        prob = [self.dqn_dict[x][3] for x in self.maps]

        for remote in self.remotes:
            selected_map = np.random.choice(self.maps, prob)
            remote.send(('reset', selected_map))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True
            
    def __len__(self):
        return self.nenvs

def initialize_dqn_dict(data_path, gamma):
    dic = {}
    if os.path.isdir(data_path):
        map_files = os.listdir(data_path)
        N = len(map_files)
        for map_file in map_files:
            w = 1
            h = 0
            r = 0
            p = (1-gamma)*(w/N) + (gamma/N)
            dic[map_file] = (h, r, w, p)
    else:
        raise ValueError('data path is not correct, plz check it.')
    return dic, N
