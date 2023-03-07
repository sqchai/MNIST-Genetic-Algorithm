# code modified from https://github.com/pytorch/examples/blob/main/mnist/main.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm


from evonet import EvoNet, evo_w


def test(pi, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    acc = 100. * correct / len(test_loader.dataset)
    print('\nmodel {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        pi,
        test_loss, correct, len(test_loader.dataset),
        acc))
    
    return acc

def test2(models, device, test_loader, train=False):
    for model in models:
        model.eval()
    
    N_models = len(models)
    test_loss = np.zeros(N_models)
    correct = np.zeros(N_models)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            for mi, model in enumerate(tqdm(models)):
                output = model(data)
                test_loss[mi] += F.cross_entropy(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct[mi] += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    if train:
        return acc
    for mi in range(N_models):
        print('model {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            mi,
            test_loss[mi], 
            correct[mi], 
            len(test_loader.dataset),
            acc[mi]
            ))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=4000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    device = 'cuda'
    use_cuda = True

    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 12,
                       'pin_memory': True,
                       'shuffle': True}
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **test_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    POP = 10000
    SUR = 400
    models = []
    survivors = []
    # betas = np.linspace(0.05, 0, args.epochs)

    for epoch in range(1, args.epochs + 1):
        print('\nEpoch: {} / {}'.format(epoch, args.epochs))
        # give birth to new models
        print('\ngenerating:')
        if epoch == 1:
            for _ in range(POP):
                # init a new model
                models.append(EvoNet().to(device))
        else:
            models = []
            # beta = max(0.02, betas[epoch - 1])
            beta = 0.035
            print(beta)
            for parent in survivors:
                for _ in range(int(POP / SUR)):
                    child = EvoNet().to(device)
                    evo_w(child, parent, beta)
                    models.append(child)

        # each model live a life by testing on training set
        print('\nliving:')
        pop_accs = test2(models, device, train_loader, train=True)
        
        # bad models die
        sorted_models = []
        ids = np.argsort(pop_accs)[::-1]  # acc high to low
        for id in ids:
            sorted_models.append(models[int(id)])
        survivors = sorted_models[:SUR]
        

        # eval models
        print('\ntesting:')
        _ = test2(survivors, device, test_loader, train=False)

    if args.save_model:
        # for mi, mm in enumerate(models):
        #     torch.save(mm.state_dict(), "mnist_cnn_{}.pt".format(mi))
        torch.save(survivors[0].state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()