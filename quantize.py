import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, STL10
from tqdm import tqdm
import wandb

torch.backends.quantized.engine = 'x86'

from torch.ao.quantization import (
  default_dynamic_qconfig,
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx
import copy

import utils
from utils import train_val
from model import Model, Classifier, TwoLayerClassifier


def calibrate(model, loader):
    model.eval()
    bar = tqdm(loader)
    with torch.no_grad():
        for image, target in bar:
            image = image.cpu()
            image = image.cuda()
            model(image)

def quantize(net: nn.Module, test_loader):
    model_to_quantize = copy.deepcopy(net)
    # qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    qconfig_mapping = get_default_qconfig_mapping("x86")
    # qconfig_mapping = QConfigMapping().set_global(default_dynamic_qconfig)
    model_to_quantize = model_to_quantize.cpu()
    model_to_quantize.eval()
    # prepare
    model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, next(iter(test_loader))[0])

    # calibrate (not shown)
    model_prepared = model_prepared.cuda()
    calibrate(model_prepared, test_loader)
    model_prepared = model_prepared.cpu()

    # quantize
    model_quantized = quantize_fx.convert_fx(model_prepared)
    # print(model_quantized.conv1.weight().dtype)
    save_quantize_model(model_quantized)
    return model_quantized

def save_quantize_model(model_quantized, path: str='static_quantize.pth'):
    torch.save(model_quantized.state_dict(), path)

def load_quantize_model(net, path: str, example_inputs):
    qconfig_mapping = get_default_qconfig_mapping("x86")
    net.eval()
    # prepare
    model_prepared = quantize_fx.prepare_fx(net, qconfig_mapping, example_inputs)
    # quantize
    model_quantized = quantize_fx.convert_fx(model_prepared)
    model_quantized.load_state_dict(torch.load(path, weights_only=True))

    return model_quantized


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of images in each mini-batch')
    parser.add_argument('--arch', default='one', type=str, help='Specify CLS Architecture one or two')
    parser.add_argument('--seed', default=42, type=int, help='specify static random seed')
    parser.add_argument('--dataset', default='stl10', type=str, help='Training Dataset (e.g. CIFAR10, STL10)')
    parser.add_argument('--quantization_method', default='half', type=str, help='quantization method. (e.g. int8, half)')
    parser.add_argument('--wandb_model_runpath', default='', type=str, help='the runpath if using a model stored in WandB')
    parser.add_argument('--wandb_project', default='default_project', type=str, help='WandB Project name')
    parser.add_argument('--model_path', type=str, default='results/128_4096_0.5_0.999_200_256_500_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--wandb_run', default='default_run', type=str, help='WandB run name')

    args = parser.parse_args()
    model_path, batch_size = args.model_path, args.batch_size

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
        "arch": "resnet34",
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "model": model_path,
        "arch": args.arch,
        "seed": args.seed,
        "method": args.quantization_method,
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

    # model setup and optimizer config
    if args.wandb_model_runpath != '':
        import os
        if os.path.exists(args.model_path):
            os.remove(args.model_path)
        base_model = wandb.restore(args.model_path, run_path=args.wandb_model_runpath)
        model_path = base_model.name
    model.load_state_dict(torch.load(model_path), strict=False)

    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    epoch = 1
    epochs = 1
    example_inputs = next(iter(test_loader))[0].cuda()

    # accuracy before quantization
    print(model(example_inputs))
    btest_loss, btest_acc_1, btest_acc_5 = train_val(model, test_loader, None, loss_criterion, epoch, epochs, 'cuda')

    # quantization
    # model = model.half()
    model = quantize(model, test_loader)

    # accuracy after quantization
    print(model(example_inputs.cpu()))
    atest_loss, atest_acc_1, atest_acc_5 = train_val(model, test_loader, None, loss_criterion, epoch, epochs, 'cpu')

    # wandb.log({'after_test_loss': atest_loss, 'after_test_acc@1': atest_acc_1, 'after_test_acc@5': atest_acc_5, 'before_test_loss': btest_loss, 'before_test_acc@1': btest_acc_1, 'before_test_acc@5': btest_acc_5})
    log_text = {'before': {'test_loss': btest_loss, 'test_acc@1': btest_acc_1, 'test_acc@5': btest_acc_5}, 'after': {'test_loss': atest_loss, 'test_acc@1': atest_acc_1, 'test_acc@5': atest_acc_5}, 'acc_diff': btest_acc_1-atest_acc_1, 'acc_diff@5': btest_acc_5-atest_acc_5}
    wandb.log(log_text)

    torch.save(model.state_dict(), f"results/quant_{args.quantization_method}_model.pth")

    wandb.save(f"results/quant_{args.quantization_method}_model.pth")
    wandb.finish()


