import torch
import numpy as np
from skimage.transform import resize

def hwc2chw(states, test=False):
    if test:
        states = downScale(states)
        return torch.tensor(np.transpose(states, (2, 0, 1)), dtype=torch.float32)
    x = [torch.tensor(np.transpose(downScale(state), (2, 0, 1)), dtype=torch.float32).unsqueeze(0) for state in states]
    #x = [torch.tensor(np.transpose(state, (2, 0, 1)), dtype=torch.float32).unsqueeze(0) for state in states]
    return torch.cat(x, 0)

def downScale(state):
    #return 255 * resize(state, (80, 80, 3))
    return resize(state, (80, 80, 3))
