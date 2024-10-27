import argparse
import os
import random
import csv
from scipy.stats import kstest

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import copy
import matplotlib.pyplot as plt

import utils
from model import Model, Classifier, StudentModel
from quantize import load_quantize_model

def sim(model, memory_data_loader, test_data_loader, num_of_samples:int=500, encoder_flag:bool=True, device:str='cuda'):
    model.eval()
    similarity = torch.nn.CosineSimilarity(dim=1)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    test_feature_bank, train_feature_bank = [], []
    test_var, train_var = [], []
    idx_09 = []
    counter = 0
    test_counter = 0
    total_correct_1, test_total_correct_1 = 0, 0
    with torch.no_grad():
        # generate feature bank
        for x, target in tqdm(memory_data_loader, desc='Feature extracting'):
            target = target.to(device)
            if counter > num_of_samples:
                break
            if counter == 0:
                fig, axes = plt.subplots(3, 3, tight_layout=True)
                axes[0, 0].axis("off")
                axes[0, 1].axis("off")
                axes[0, 2].axis("off")
                axes[1, 0].axis("off")
                axes[1, 1].axis("off")
                axes[1, 2].axis("off")
                axes[2, 0].axis("off")
                axes[2, 1].axis("off")
                axes[2, 2].axis("off")
                axes[0, 0].imshow(x[0][1].permute(1, 2, 0))
                axes[0, 1].imshow(x[1][1].permute(1, 2, 0))
                axes[0, 2].imshow(x[2][1].permute(1, 2, 0))
                axes[1, 0].imshow(x[3][1].permute(1, 2, 0))
                axes[1, 1].imshow(x[4][1].permute(1, 2, 0))
                axes[1, 2].imshow(x[5][1].permute(1, 2, 0))
                axes[2, 0].imshow(x[6][1].permute(1, 2, 0))
                axes[2, 1].imshow(x[7][1].permute(1, 2, 0))
                axes[2, 2].imshow(x[8][1].permute(1, 2, 0))
                fig.savefig('results/seed_check.png')
                plt.close()
            feature_list = []
            for data in x:
                if encoder_flag:
                    f, g = model(data.to(device, non_blocking=True))
                else:
                    f = model(data.to(device, non_blocking=True))
                feature_list.append(f)

            cos_list = []
            for i in range(len(x)):
                for j in range(len(x)):
                    if i >= j:
                        continue
                    cos_list.append(similarity(feature_list[i], feature_list[j]))

            var = torch.var(torch.stack(cos_list, dim=1), dim=1)
            train_var.append(var)
            result = cos_list[0]
            for i in range(1,len(cos_list)):
                result += cos_list[i]
            result /= len(cos_list)

            for i in range(len(result)):
                if result[i] >= 0.94:
                    idx_09.append(i+counter)

            train_feature_bank.append(result)
            counter += len(result)

        counter=0
        for x, target in tqdm(test_data_loader, desc='Feature extracting'):
            target = target.to(device)
            if counter > num_of_samples:
                break
            feature_list = []
            for data in x:
                if encoder_flag:
                    f, g = model(data.to(device, non_blocking=True))
                else:
                    f = model(data.to(device, non_blocking=True))
                feature_list.append(f)

            cos_list = []
            for i in range(len(x)):
                for j in range(len(x)):
                    if i >= j:
                        continue
                    cos_list.append(similarity(feature_list[i], feature_list[j]))
            var = torch.var(torch.stack(cos_list, dim=1), dim=1)
            test_var.append(var)
            result = cos_list[0]
            for i in range(1,len(cos_list)):
                result += cos_list[i]
            result /= len(cos_list)

            test_feature_bank.append(result)
            counter += len(result)

        # [D, N]
        test_feature_bank = torch.cat(test_feature_bank, dim=0).contiguous()
        train_feature_bank = torch.cat(train_feature_bank, dim=0).contiguous()
        train_var = torch.cat(train_var, dim=0)
        print('Accuracy model dataset:', total_correct_1/counter)

    color = ['tab:blue', 'tab:orange', 'tab:green']

    train_random_sampling = random.sample(range(0, len(train_feature_bank)), num_of_samples)
    test_random_sampling = random.sample(range(0, len(test_feature_bank)), num_of_samples)
    olabels = ['train', 'test']
    # data = [train_feature_bank[train_random_sampling].to('cpu').detach().numpy().copy(),test_feature_bank[test_random_sampling].to('cpu').detach().numpy().copy()]
    data = [train_feature_bank[:num_of_samples].to('cpu').detach().numpy().copy(),test_feature_bank[:num_of_samples].to('cpu').detach().numpy().copy()]
    ks_result = kstest(data[0], data[1], alternative='two-sided', method='auto')
    # plt.title(f'{num_of_samples}_{ks_result.pvalue}')
    plt.title(f'train & test similarity distribution, {num_of_samples} samples')
    plt.hist(data[0], 30, alpha=0.6, density=False, label=olabels[0], stacked=False, range=(0.4, 1.0), color=color[0])
    plt.hist(data[1], 30, alpha=0.6, density=False, label=olabels[1], stacked=False, range=(0.4, 1.0), color=color[1])
    plt.legend()
    plt.ylabel('The number of samples')
    plt.xlabel('Mean Cosine Similarity')
    plt.savefig(f"results/sim_test_train_model.png")
    plt.close()

    data = [train_feature_bank.to('cpu').detach().numpy().copy(),test_feature_bank.to('cpu').detach().numpy().copy()]
    ks_result_all = kstest(train_feature_bank.to('cpu').detach().numpy().copy(),test_feature_bank.to('cpu').detach().numpy().copy(), alternative='two-sided', method='auto')
    # plt.title(f'all_{ks_result_all.pvalue}')
    plt.title(f'train & test similarity distribution, {num_of_samples} samples, {ks_result_all.pvalue}')
    plt.hist(data[0], 30, alpha=0.6, density=False, label=olabels[0], stacked=False, range=(0.4, 1.0), color=color[0])
    plt.hist(data[1], 30, alpha=0.6, density=False, label=olabels[1], stacked=False, range=(0.4, 1.0), color=color[1])
    plt.legend()
    plt.ylabel('The number of samples')
    plt.xlabel('Mean Cosine Similarity')
    plt.savefig("results/sim_test_train_model_all.png")
    plt.close()

    return ks_result.pvalue, ks_result.statistic

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoCo')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for each image')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--classes', default=10, type=int, help='the number of classes')
    parser.add_argument('--num_of_samples', default=2500, type=int, help='num of samples')
    parser.add_argument('--dataset', default='stl10', type=str, help='Training Dataset (e.g. CIFAR10, STL10)')
    parser.add_argument('--seed', default=42, type=int, help='specify static random seed')
    parser.add_argument('--model_path', type=str, default='results/128_4096_0.5_0.999_200_256_500_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--wandb_model_runpath', default='', type=str, help='the runpath if using a model stored in WandB')
    parser.add_argument('--wandb_project', default='default_project', type=str, help='WandB Project name')
    parser.add_argument('--wandb_run', default='default_run', type=str, help='WandB run name')
    parser.add_argument('--is_encoder', action='store_true', help='is model encoder?')
    parser.add_argument('--is_modification', action='store_true', help='is model modificated?')
    parser.add_argument('--model_modification_method', default='', type=str, help='is is_modification, specify model modification method(e.g. quant, prune, distill)')

    # args parse
    args = parser.parse_args()
    batch_size = args.batch_size

    # initialize random seed
    utils.set_random_seed(args.seed)

    # wandb init
    config = {
        "arch": "resnet34",
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "num_of_samples": args.num_of_samples,
        "seed": args.seed,
        "is_encoder": args.is_encoder,
        "is_modification": args.is_modification,
        "modification_method": args.model_modification_method,
    }
    wandb.init(project=args.wandb_project, name=args.wandb_run, config=config)

    # data prepare
    if args.dataset == 'stl10':
        memory_data = utils.STL10NAug(root='data', split='unlabeled', transform=utils.stl_train_transform, download=True)
        memory_data.set_mia_train_dataset_flag(True)
        test_data = utils.STL10NAug(root='data', split='unlabeled', transform=utils.stl_train_transform, download=True)
        test_data.set_mia_train_dataset_flag(False)
    elif args.dataset == 'cifar10':
        memory_data = utils.CIFAR10NAug(root='data', train=True, transform=utils.train_transform, download=True, n=10)
        test_data = utils.CIFAR10NAug(root='data', train=False, transform=utils.train_transform, download=True, n=10)
    else:
        memory_data = utils.CIFAR100NAug(root='data', train=True, transform=utils.train_transform, download=True, n=10)
        test_data = utils.CIFAR100NAug(root='data', train=False, transform=utils.train_transform, download=True, n=10)
    shuffle=True
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)

    if not os.path.exists('results'):
        os.mkdir('results')
    # model setup and optimizer config
    if args.wandb_model_runpath != '':
        import os
        if os.path.exists(args.model_path):
            os.remove(args.model_path)
        base_model = wandb.restore(args.model_path, run_path=args.wandb_model_runpath)
        model_path = base_model.name

    device = 'cuda'
    # model setup and optimizer config
    if args.is_encoder:
        model = Model(args.feature_dim).cuda()
        model.load_state_dict(torch.load(model_path), strict=False)
    elif args.is_modification and args.model_modification_method == 'quant':
        model = Classifier(args.classes).cpu()
        model = load_quantize_model(model, model_path, torch.randn((3,4,32,32)))
        device = 'cpu'
    elif args.is_modification and args.model_modification_method == 'distill':
        model = StudentModel(args.classes, 'mobilenet_v2').cuda()
        model.load_state_dict(torch.load(model_path))
    elif args.is_modification and args.model_modification_method == 'prune':
        model = Classifier(args.classes).cuda()
        model.load_state_dict(torch.load(model_path))
    else:
        model = Classifier(args.classes).cuda()
        model.load_state_dict(torch.load(model_path), strict=False)

    pvalue, statistic = sim(model, memory_loader, test_loader, num_of_samples=args.num_of_samples, encoder_flag=args.is_encoder, device=device)
    # sim(model_q, memory_loader, memory_loader)

    # save kstest result
    wandb.log({'pvalue': pvalue, 'statistic': statistic})

    # wandb finish
    os.remove(os.path.join(wandb.run.dir, args.model_path))
    wandb.save("results/seed_check.png")
    wandb.save("results/sim_test_train_model.png")
    wandb.save("results/sim_test_train_model_all.png")
    wandb.finish()


