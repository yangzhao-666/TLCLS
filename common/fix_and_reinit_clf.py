import torch
import pickle

from common.classifier import Classifier

def load_model_and_fine_tune_prediction(model_path, actor_critic, optimizer):
    print('model already exists, loading the model from prediction....')

    clf = Classifier((3, 80, 80), 49)
    clf = clf.cuda()
    checkpoint = torch.load(model_path + 'clf.pt')
    clf.load_state_dict(checkpoint['clf'])

    for i in range(3):
        index = i * 2
        actor_critic.features[index].weight = clf.features[index].weight
    
    actor_critic.train()

    for child in actor_critic.children():
        for i, param in enumerate(child.parameters()):
            if i > 1:
                break
            param.requires_grad = False
        break
    
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(actor_critic.features[2].weight)
        torch.nn.init.zeros_(actor_critic.features[2].bias)
        torch.nn.init.xavier_uniform_(actor_critic.features[4].weight)
        torch.nn.init.zeros_(actor_critic.features[4].bias)
        torch.nn.init.xavier_uniform_(actor_critic.fc[0].weight)
        torch.nn.init.zeros_(actor_critic.fc[0].bias)
        torch.nn.init.xavier_uniform_(actor_critic.actor.weight)
        torch.nn.init.zeros_(actor_critic.actor.bias)
        torch.nn.init.xavier_uniform_(actor_critic.critic.weight)
        torch.nn.init.zeros_(actor_critic.critic.bias)
    print('already loaded the first feature extraction layer from prediction tasks and will fine-tune the remaining layers...')

    print('loaded successfully, will train continually')

    return actor_critic, optimizer
