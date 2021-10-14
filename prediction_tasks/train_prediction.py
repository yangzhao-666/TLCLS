import os

import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np

from classifier import Classifier
from data_utils import sampler

import matplotlib.pyplot as plt

def train(num_epoch, num_iter, batch_size, data_sampler, clf, criterion, optimizer, i_run):

    accs = []
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i in range(num_iter):
                      # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data_sampler.sample()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = clf(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 0:
                            #print('[%d, %5d] loss: %.3f' %
                                                    #      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                y_pre = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                acc = np.sum(y_pre == labels.detach().cpu().numpy()) / len(labels)
                print('run: {} | epoch: {} | iter: {} | acc: {}'.format(i_run, epoch, i, acc))
                accs.append(acc)
    #print('Finished Training')
    return accs

def train_prediction_agent(sampler, num_epoch=10, num_iter=100, batch_size=32, data_n_boxes=1):

    #specify data path, which is the same as the path where the model will be saved;
    if data_n_boxes == 3:
        data_dir = '/3_boxes'
    elif data_n_boxes == 2:
        data_dir = '/2_boxes'
    elif data_n_boxes == 1:
        data_dir = '/1_box'
    else:
        raise ValueError('please check data path')

    data_sampler = sampler
    accs = []
    for i in range(5):
        save_path = '../results/entropy_coef_0.1/' + data_dir + '_prediction/' + str(i+1) + '/pre_train/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        clf = Classifier((3, 80, 80), 49)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(clf.parameters(), lr=0.001, momentum=0.9)

        acc = train(num_epoch, num_iter, batch_size, data_sampler, clf.cuda(), criterion, optimizer, i)
        accs.append(acc)
        #save the model
        torch.save(
                    {
                        'clf': clf.state_dict()
                    },
                        save_path + 'clf.pt'
                    )

    avg = np.mean((np.array(accs[0]), np.array(accs[1]), np.array(accs[2]), np.array(accs[3]), np.array(accs[4])), axis=0)
    var = np.std((np.array(accs[0]), np.array(accs[1]), np.array(accs[2]), np.array(accs[3]), np.array(accs[4])), axis=0)
    ci = 1.96 * var/np.sqrt(5)
    ev_up = avg-ci
    ev_down = avg+ci
    return avg, ev_up, ev_down


if __name__ == '__main__':
    data_path = './data_1000.pt'
    num_epoch = 10
    num_iter = 100
    batch_size = 256
    data_n_boxes = 2

    data = torch.load(data_path)
    x = data['x']
    y = data['y']
    sampler = sampler(x, y, batch_size)
    plot_1 = train_prediction_agent(sampler, num_epoch=10, num_iter=100, batch_size=32, data_n_boxes=1)
    
    #plot and save the plot

    cmaps = ['blue', 'red', 'grey']
    labels = ['training prediction on 1-box dataset']
    fig, ax = plt.subplots(figsize=(15, 10))
    resutls = [plot_1]
    for n, p in enumerate(resutls):
        x = [i for i in range(len(p[0]))]
        ax.plot(x, p[0], color=cmaps[n], label=labels[n])
        ax.fill_between(x, p[1], p[2], color=cmaps[n], alpha=.1)
        ax.legend(loc='lower right', fontsize=15)
    plt.savefig('./figure.png')
