from common.utils import hwc2chw

import random
import torch
import numpy as np

class sampler():

    def __init__(self, x, y, batch_size):
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.len = len(x)
        if self.len < batch_size:
            raise ValueError('data size is smaller than batch size')
        
    def sample(self):
        index = np.random.choice(range(self.len), self.batch_size, replace=False)
        return self.x[index].cuda(), self.y[index].cuda()

#get n_exp episodes of data, each episode is played by a trained model;
def get_agent_data(env, n_exp, agent, epsilon=0.1):
    
    actions = torch.tensor((0, 1, 2, 3, 4))

    state = env.reset()
    x = hwc2chw(state, test=True).unsqueeze(0)
    ply_pos = env.player_position
    y = ply_pos[0] * 7 + ply_pos[1]
    y = torch.tensor(y, dtype=torch.long).unsqueeze(0)
    
    for i in range(n_exp):
        print('generating data of {} th episode...'.format(i+1))
        state = env.reset()
        state = hwc2chw(state, test=True).unsqueeze(0)
        ply_pos = env.player_position
        ply_pos = ply_pos[0] * 7 + ply_pos[1]
        ply_pos = torch.tensor(ply_pos, dtype=torch.long).unsqueeze(0)
        x = torch.cat((x, state))
        y = torch.cat((y, ply_pos))
    
        radn = random.random()
        if radn < epsilon:
            action = random.choice(actions)
        else:
            action = agent.select_action(state)

        next_state, _, done, _ = env.step(action.item())
        state = hwc2chw(next_state, test=True).unsqueeze(0)
        ply_pos = env.player_position
        ply_pos = ply_pos[0] * 7 + ply_pos[1]
        ply_pos = torch.tensor(ply_pos, dtype=torch.long).unsqueeze(0)
        x = torch.cat((x, state))
        y = torch.cat((y, ply_pos))
        while not done:
            radn = random.random()
            if radn < epsilon:
                action = random.choice(actions)
            else:
                action = agent.select_action(state)

            next_state, _, done, _ = env.step(action.item())
            state = hwc2chw(next_state, test=True).unsqueeze(0)
            ply_pos = env.player_position
            ply_pos = ply_pos[0] * 7 + ply_pos[1]
            ply_pos = torch.tensor(ply_pos, dtype=torch.long).unsqueeze(0)
            x = torch.cat((x, state))
            y = torch.cat((y, ply_pos))
    data = {'x': x, 'y':y}
    torch.save(data, './data_1000.pt')
