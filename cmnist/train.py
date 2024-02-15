import random
import numpy as np
import argparse
import os
import string
import random
import shutil

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import utils as TVutils

from tqdm import tqdm

import pathlib
from typing import List, Tuple, Union, Any, Optional, Sequence

from datasets import CMNISTPairedDataset

# data parameters
INP_CHANS = 6
INP_SPAT = (16, 16)

# training parmeters
BATCH_SIZE = 128
LR = 1e-4

# OT parameters
COST_COEFFICIENT = 0.0

# ebm training parameters
ALPHA = 0.001
ENERGY_SAMPLING_STEP = 1.0
ENERGY_SAMPLING_ITERATIONS = 100
LANGEVIN_SAMPLING_NOISE = 0.025

# sample buffer setup
SAMPLE_BUFFER_TYPE = 'static'
SAMPLE_BUFFER_PARAMS = {
    'static': {
        'size': 10000
    },
    'igebm': {
        'size': 10000,
        'p': 0.95
    }
}


# basic paths
THIS_FILE_DIR = pathlib.Path(__file__).parent.resolve()
BASIC_FILE_DIR = '/home/pvmokrov/egeot_joint_project/igebm-pytorch' # THIS_FILE_DIR

##############
# argparse

parser = argparse.ArgumentParser(
    description='EgEOT lab',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--directory', type=str, default='rand')
parser.add_argument('-hw', '--hardware', action='store', type=str, default='cuda:0')

ARGPARSE_ARGS = parser.parse_args()

###############
# device

if ARGPARSE_ARGS.hardware.startswith('cuda'):
    DEVICE = 'cuda'
    torch.cuda.set_device(int(ARGPARSE_ARGS.hardware.split(':')[1]))
else:
    DEVICE = 'cpu'

###############
# directories to save the results

DATASET_DATA_PATH = os.path.join(BASIC_FILE_DIR, 'data/cmnist')
EXP_DIR = './'
FILES_TO_MIGRATE = ['train.py', 'datasets.py']

def save_code():
    def save_file(file_name):
        file_in = open(os.path.join('./', file_name), 'r')
        file_out = open(os.path.join(EXP_DIR, os.path.basename(file_name)), 'w')
        for line in file_in:
            file_out.write(line)
    for file in FILES_TO_MIGRATE:
        save_file(file)

if ARGPARSE_ARGS.directory == 'this':
    #TODO: check this regime!
    this_items_to_remove = [f for f in os.listdir(THIS_FILE_DIR) if not f in FILES_TO_MIGRATE]
    for item in this_items_to_remove:
        item_path = os.path.join(THIS_FILE_DIR, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            raise Exception('Impossible exception')
    EXP_DIR = THIS_FILE_DIR
else:
    if ARGPARSE_ARGS.directory == 'rand':
        dir_name =  'exp_' + ''.join(random.choices(string.ascii_lowercase, k=10))
    else:
        dir_name = ARGPARSE_ARGS.directory
    EXP_DIR = os.path.join(BASIC_FILE_DIR, dir_name)
    if os.path.exists(EXP_DIR):
        # prevents overwriting old experiment folders by accident
        raise RuntimeError('Folder "{}" already exists. Please use a different "EXP_DIR".'.format(EXP_DIR))
    else:
        os.makedirs(EXP_DIR)
    save_code()

SAMPLES_DIR = os.path.join(EXP_DIR, 'samples')
os.makedirs(SAMPLES_DIR)

##############
# OT cost

def sq_cost(X : torch.Tensor, Y : torch.Tensor, coeff=COST_COEFFICIENT):
    return coeff * (X-Y).square().flatten(start_dim=1).mean(dim=1)

COST = sq_cost
    
##############
# Model

#1. Simple ConvNet with hard-coded cost function

class SimpleConvNetHard(nn.Module):

    def __init__(self):
        super().__init__()

        n_channels = 32
        self.block_joint = nn.Sequential(
            *[
                nn.Conv2d(3, n_channels, 3, padding=1, stride=1), # (16 x 16)
                nn.BatchNorm2d(n_channels),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(n_channels, n_channels * 2, kernel_size=4, padding=1, stride=2), # (8, 8)
                nn.BatchNorm2d(n_channels * 2),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=4, padding=1, stride=2), # (4, 4)
                nn.BatchNorm2d(n_channels * 4),
                nn.LeakyReLU(0.2, True),
            ]
        )

        self.fin_blocks1 = nn.Sequential(
            *[
                nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=4, padding=1, stride=2), # (2, 2)
                nn.BatchNorm2d(n_channels * 8),
                nn.LeakyReLU(0.2, True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(n_channels * 8, 1)
            ]
        )

        self.fin_blocks2 = nn.Sequential(
            *[
                nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=4, padding=1, stride=2), # (2, 2)
                nn.BatchNorm2d(n_channels * 8),
                nn.LeakyReLU(0.2, True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(n_channels * 8, 1)
            ]
        )

    def forward(self, input : torch.Tensor):
        inp1, inp2 = torch.chunk(input, 2, 1)
        out1, out2 = self.fin_blocks1(self.block_joint(inp1)), self.fin_blocks2(self.block_joint(inp2))
        # u(x) + v(y) - c(x, y)
        res = out1 + out2 - COST(inp1, inp2)

        return res

#2. ...

################
# SampleBuffers

class SampleBufferGeneric:
    
    def __init__(self):
        pass
    
    def __len__(self):
        raise NotImplementedError()
    
    def push(self, Xs, ids):
        raise NotImplementedError()
    
    def get(self, n_samples):
        raise NotImplementedError()
    
    def __call__(self, Xs : torch.Tensor) -> Tuple[torch.Tensor, Optional[Sequence]]:
        raise NotImplementedError()

class SampleBufferIGEBM(SampleBufferGeneric):

    def __init__(self, p=0.95, max_samples=10000):
        self.p = p
        self.max_samples = max_samples
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def push(self, Xs : torch.Tensor, ids : Optional[Sequence] = None):
        Xs = Xs.detach().cpu()
        
        if ids is None:
            for X in Xs:
                self.buffer.append(X)
            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

        else:
            assert max(ids) < len(self.buffer)
            assert len(ids) == len(Xs)
            for i, X in zip(ids, Xs):
                self.buffer[i] = X

    def get(self, n_samples, device='cpu'):
        indices = random.choices(range(len(self.buffer)), k=n_samples)
        Xs = [self.buffer[i] for i in indices]
        Xs = torch.stack(Xs, 0).to(device)
        return Xs, indices

    def get_random(self, Xs):
        samples = torch.rand(Xs.size, device=Xs.device)
        return samples, None
    
    def __call__(self, Xs):
        batch_size = Xs.size(0)
        if len(self) < 1:
            return self.get_random(Xs)

        n_replay = (np.random.rand(batch_size) < self.p).sum()

        if n_replay == 0:
            return self.get_random(Xs)
        if n_replay == batch_size:
            return self.get(n_replay, device=Xs.device)

        replay_samples, _ = self.get(n_replay)
        random_samples, _ = self.get_random(Xs[n_replay:])
        samples = torch.cat([replay_samples, random_samples], 0)

        return samples, None


class SampleBufferStatic(SampleBufferGeneric):
    
    def __init__(self, Xs_init):
        super().__init__()
        self.data = Xs_init.clone().detach().cpu()
    
    def __len__(self):
        return len(self.data)
    
    def push(self, Xs, ids):
        self.data[ids] = Xs.detach().cpu()
        del Xs
    
    def get(self, n_samples, device='cpu'):
        indices = np.random.choice(len(self), n_samples)
        return self.data[indices].to(device), indices
    
    def __call__(self, Xs):
        return self.get(len(Xs), device=Xs.device)

def initialize_sample_buffer() -> SampleBufferGeneric:
    if SAMPLE_BUFFER_TYPE == 'static':
        Xs_init = torch.rand(SAMPLE_BUFFER_PARAMS['static']['size'], INP_CHANS, INP_SPAT[0], INP_SPAT[1])
        return SampleBufferStatic(Xs_init)
    if SAMPLE_BUFFER_TYPE == 'igebm':
        p, max_samples = SAMPLE_BUFFER_PARAMS['igebm']['p'], SAMPLE_BUFFER_PARAMS['igebm']['size']
        return SampleBufferIGEBM(p, max_samples)
    raise Exception('Unknown sample buffer type!')
    
##########
# utils

def sample_data(loader):
    loader_iter = iter(loader)

    while True:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)

            yield next(loader_iter)


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def clip_grad(parameters, optimizer):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))

################################################################################################################
################################################################################################################
# Training

model = SimpleConvNetHard().to(DEVICE)
dataset = CMNISTPairedDataset(
    train=True, spat_dim=INP_SPAT, download=False, dummy_class=False, root=DATASET_DATA_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
loader = tqdm(enumerate(sample_data(loader)))

sample_buffer = initialize_sample_buffer()

noise = torch.randn(BATCH_SIZE, INP_CHANS, INP_SPAT[0], INP_SPAT[1], device=DEVICE)

parameters = model.parameters()
optimizer = optim.Adam(parameters, lr=LR, betas=(0.0, 0.999))

for i, pos_img in loader:
    pos_img = pos_img.to(DEVICE)
    neg_img, neg_ids = sample_buffer(pos_img)

    neg_img.requires_grad = True

    requires_grad(parameters, False)
    model.eval()

    for k in tqdm(range(ENERGY_SAMPLING_ITERATIONS)):
        if noise.shape[0] != neg_img.shape[0]:
            noise = torch.randn(neg_img.shape[0], INP_CHANS, INP_SPAT[0], INP_SPAT[1], device=DEVICE)

        noise.normal_(0, LANGEVIN_SAMPLING_NOISE)

        neg_out = model(neg_img)
        neg_out.sum().backward()
        neg_img.grad.data.clamp_(-0.01, 0.01)

        neg_img.data.add_(neg_img.grad.data, alpha=-ENERGY_SAMPLING_STEP)
        neg_img.data.add_(noise.data)

        neg_img.grad.detach_()
        neg_img.grad.zero_()

        neg_img.data.clamp_(0, 1)

    neg_img = neg_img.detach()

    requires_grad(parameters, True)
    model.train()
    model.zero_grad()

    pos_out = model(pos_img)
    neg_out = model(neg_img)

    loss = ALPHA * (pos_out ** 2 + neg_out ** 2)
    loss = loss + (pos_out - neg_out)
    loss = loss.mean()
    loss.backward()

    clip_grad(parameters, optimizer)

    optimizer.step()

    sample_buffer.push(neg_img, neg_ids)

    loader.set_description(f'loss: {loss.item():.5f}')

    if i % 50 == 0:
        neg_imgX, neg_imgY = torch.chunk(neg_img, 2, 1)
        for suff, img in zip(['X', 'Y'], [neg_imgX, neg_imgY]):
            TVutils.save_image(
                img.detach().to('cpu'),
                f'{SAMPLES_DIR}/{str(i).zfill(5)}_{suff}.png',
                nrow=16,
                normalize=True,
                range=(0, 1),
            )

