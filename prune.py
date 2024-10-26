import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, STL10
from tqdm import tqdm
import wandb

import utils
from utils import train_val
from model import Model, Classifier, TwoLayerClassifier


def get_conv1_linear_modules(net):
    pruned_modules = []
    for name, module in net.named_modules():
        # print('Check', module)
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            pruned_modules.append((module, 'weight'))
    return pruned_modules

def remove_prune_layer(pruned_modules):
    for module, id in pruned_modules:
        prune.remove(module, id)

def calculate_prune_ratio(modules):
    zero_neuron = 0
    all_neuron = 0
    for module, id in modules:
        zero_neuron += int(torch.sum(module.weight == 0))
        all_neuron += int(module.weight.nelement())
        print(
            "Sparsity in {}: {:.2f}%".format(module.__str__().split('(')[0],
                100. * float(torch.sum(module.weight == 0))
                / float(module.weight.nelement())
            )
        )
    print('pruning ratio:', (zero_neuron/all_neuron) * 100.0)
    return (zero_neuron/all_neuron) * 100.0

def pruning(net: nn.Module, pruned_modules, ratio: float=0.5, interactive_pruning: bool=False, iter: int=3):
    if interactive_pruning:
        ratio = ratio ** (1.0/iter)
    else:
        iter = 1

    print('interactive_pruning is not implemented')
    prune.global_unstructured(
        pruned_modules,
        pruning_method=prune.L1Unstructured,
        amount=ratio,
    )

    return net

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='results/128_4096_0.5_0.999_200_256_500_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning Rate at the training start')
    parser.add_argument('--prune_rate', default=0.30, type=float, help='Pruning rate')
    parser.add_argument('--arch', default='one', type=str, help='Specify CLS Architecture one or two')
    parser.add_argument('--seed', default=42, type=int, help='specify static random seed')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='Weight Decay')
    parser.add_argument('--dataset', default='stl10', type=str, help='Training Dataset (e.g. CIFAR10, STL10)')
    parser.add_argument('--wandb_model_runpath', default='', type=str, help='the runpath if using a model stored in WandB')
    parser.add_argument('--wandb_project', default='default_project', type=str, help='WandB Project name')
    parser.add_argument('--wandb_run', default='default_run', type=str, help='WandB run name')

    args = parser.parse_args()
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    lr, weight_decay = args.lr, args.weight_decay

    # initialize random seed
    utils.set_random_seed(args.seed)

    if args.wandb_model_runpath != '':
        import os
        if os.path.exists(args.model_path):
            os.remove(args.model_path)
        base_model = wandb.restore(args.model_path, run_path=args.wandb_model_runpath)
        model_path = base_model.name

    # wandb init
    config = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "arch": "resnet34",
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "model": model_path,
        "arch": args.arch,
        "seed": args.seed,
        "prune_rate": args.prune_rate,
        "wandb_model_runpath": args.wandb_model_runpath
    }
    wandb.init(project=args.wandb_project, name=args.wandb_run, config=config)

    if args.dataset == 'stl10':
        train_data = STL10(root='data', split='train', transform=utils.stl_train_ds_transform, download=True)
        test_data = STL10(root='data', split='test', transform=utils.stl_test_ds_transform, download=True)
    elif args.dataset == 'cifar10':
        train_data = CIFAR10(root='data', train=True, transform=utils.train_ds_transform, download=True)
        test_data = CIFAR10(root='data', train=False, transform=utils.test_transform, download=True)
    else:
        train_data = CIFAR100(root='data', train=True, transform=utils.train_ds_transform, download=True)
        test_data = CIFAR100(root='data', train=False, transform=utils.test_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if args.arch == 'one':
        print('CLS Architecture is specified a One Layer')
        model = Classifier(num_class=len(train_data.classes), pretrained_path=model_path).cuda()
    else:
        print('CLS Architecture is specified Two Layers')
        model = TwoLayerClassifier(num_class=len(train_data.classes), pretrained_path=model_path).cuda()
    for param in model.f.parameters():
        param.requires_grad = False
    model.load_state_dict(torch.load(model_path), strict=False)

    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    epoch = 1
    btest_loss, btest_acc_1, btest_acc_5 = train_val(model, test_loader, None, loss_criterion, epoch, epochs, 'cuda')

    # pruning
    pruned_modules = get_conv1_linear_modules(model)
    model = pruning(model, pruned_modules, args.prune_rate)
    calculate_prune_ratio(pruned_modules)
    optimizer = optim.Adam(model.fc.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # fine-tuning all param or last layer

    epoch = 1
    ptest_loss, ptest_acc_1, ptest_acc_5 = train_val(model, test_loader, None, loss_criterion, epoch, epochs, 'cuda')

    print('fine-tuning after pruning')
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer, loss_criterion, epoch, epochs, 'cuda')
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None, loss_criterion, epoch, epochs, 'cuda')
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        wandb.log({'train_loss': train_loss, 'train_acc@1': train_acc_1, 'train_acc@5': train_acc_5, 'test_loss': test_loss, 'test_acc@1': test_acc_1, 'test_acc@5': test_acc_5})
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/pruned_finetune_statistics.csv', index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'results/pruned_model.pth')

    model.load_state_dict(torch.load('results/pruned_model.pth'))
    remove_prune_layer(pruned_modules)
    calculate_prune_ratio(pruned_modules)
    torch.save(model.state_dict(), 'results/pruned_model.pth')

    epoch=1
    atest_loss, atest_acc_1, atest_acc_5 = train_val(model, test_loader, None, loss_criterion, epoch, epochs, 'cuda')
    log_text = {'before': {'test_loss': btest_loss, 'test_acc@1': btest_acc_1, 'test_acc@5': btest_acc_5}, 'after': {'test_loss': atest_loss, 'test_acc@1': atest_acc_1, 'test_acc@5': atest_acc_5}, 'pruned': {'test_loss': ptest_loss, 'test_acc@1': ptest_acc_1, 'test_acc@5': ptest_acc_5}, 'acc_diff': btest_acc_1-atest_acc_1, 'acc_diff@5': btest_acc_5-atest_acc_5}
    wandb.log(log_text)

    wandb.save('results/pruned_model.pth')
    wandb.finish()

