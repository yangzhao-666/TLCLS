import torch
import pickle

def fix_and_reinit(actor_critic, optimizer, fix='conv'):

    actor_critic.train()
    
    if fix != 'conv':
        fix = int(fix)
        for child in actor_critic.children():
            for i, param in enumerate(child.parameters()):
                if i > (2 * fix - 1):
                    break
                param.requires_grad = False
            break
        
        with torch.no_grad():
            if fix == 1:
                reinit_list = [2, 4]
            elif fix == 2:
                reinit_list = [4]
            elif fix == 3:
                reinit_list = []

            for i in reinit_list:
                torch.nn.init.xavier_uniform_(actor_critic.features[i].weight)
                torch.nn.init.zeros_(actor_critic.features[i].bias)

            torch.nn.init.xavier_uniform_(actor_critic.fc[0].weight)
            torch.nn.init.zeros_(actor_critic.fc[0].bias)
            torch.nn.init.xavier_uniform_(actor_critic.actor.weight)
            torch.nn.init.zeros_(actor_critic.actor.bias)
            torch.nn.init.xavier_uniform_(actor_critic.critic.weight)
            torch.nn.init.zeros_(actor_critic.critic.bias)
    elif fix == 'conv':
        for i, child in enumerate(actor_critic.children()):
            if i > 0:
                for param in child.parameters():
                    param.requires_grad = False
        
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(actor_critic.features[0].weight)
            torch.nn.init.zeros_(actor_critic.features[0].bias)
            torch.nn.init.xavier_uniform_(actor_critic.features[2].weight)
            torch.nn.init.zeros_(actor_critic.features[2].bias)
            torch.nn.init.xavier_uniform_(actor_critic.features[4].weight)
            torch.nn.init.zeros_(actor_critic.features[4].bias)

    return actor_critic, optimizer
