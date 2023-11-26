'''
Train a seqboat model on sequential CIFAR10 / sequential MNIST with PyTorch for demonstration purposes.

This code borrows heavily from https://github.com/HazyResearch/safari and https://github.com/kuangliu/pytorch-cifar and is based on https://github.com/HazyResearch/state-spaces.

We don't need custom learning rates for the MD-EMA based SSM.

* Train standard sequential CIFAR:
    python standalone_cifar.py
* Train sequential CIFAR grayscale:
    python standalone_cifar.py --grayscale

The default CIFAR10 SeqBoat model with prenorm flag turned on should get
85+% accuracy on the CIFAR10 val set.

'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from tqdm.auto import tqdm

from standalone_seqboat import SeqBoatModel



class LinearDecaySchedule():
    """Decay the LR on a linear schedule.
    """

    def __init__(self, args, optimizer):
        self.args = args
        self.optimizer = optimizer
        warmup_end_lr = args.lr
        if args.warmup_updates < 0:
            raise ValueError('warm up steps cannot be negative.')
        elif args.warmup_updates == 0:
            assert args.warmup_init_lr < 0
            args.warmup_init_lr = warmup_end_lr
        else:
            assert args.warmup_init_lr < warmup_end_lr
            if args.warmup_init_lr < 0:
                args.warmup_init_lr = 0

        # linearly warmup for the first args.warmup_updates
        if args.warmup_updates > 0:
            self.warmup_power = args.warmup_power
            self.warmup_factor = (warmup_end_lr - args.warmup_init_lr) / (args.warmup_updates ** args.warmup_power)
        else:
            self.warmup_power = 1
            self.warmup_factor = 0

        self.end_learning_rate = args.end_learning_rate
        self.total_num_update = args.total_num_update
        self.lr_factor = (warmup_end_lr - self.end_learning_rate) / (self.total_num_update - args.warmup_updates)

        # initial learning rate
        self.lr = args.warmup_init_lr

        self.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--warmup-updates', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-power', default=1, type=int, metavar='N', help='the power of warmup')
        parser.add_argument('--warmup-init-lr', default=0, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')
        parser.add_argument('--end-learning-rate', default=0.0, type=float)
        parser.add_argument('--total-num-update', default=1000000, type=int)

    def state_dict(self):
        return {'lr': self.lr}
    
    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def load_state_dict(self, state_dict):
        if 'lr' in state_dict:
            self.lr = state_dict['lr']

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates <= self.args.warmup_updates:
            self.lr = self.args.warmup_init_lr + (num_updates ** self.warmup_power) * self.warmup_factor
        elif num_updates >= self.total_num_update:
            self.lr = self.end_learning_rate
        else:
            self.lr = self.lr_factor * (self.total_num_update - num_updates) + self.end_learning_rate

        self.set_lr(self.lr)
        return self.lr


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Optimizer
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.02, type=float, help='Weight decay')
# Scheduler
parser.add_argument('--epochs', default=200, type=float, help='Training epochs')
# Dataset
parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=50, type=int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=8, type=int, help='Number of layers')
parser.add_argument('--d_model', default=160, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.1, type=float, help='Model Dropout')
parser.add_argument('--init_temp_scale', default=1.0, type=float, help='Initial temperature scaling factor for latent configurator')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')

LinearDecaySchedule.add_args(parser)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print(f'==> Preparing data..')

def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val

if args.grayscale:
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
        transforms.Lambda(lambda x: x.view(1, 1024).t())
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x.view(3, 1024).t())
    ])

# Train with no data augmentation
transform_train = transform_test = transform

trainset = torchvision.datasets.CIFAR10(
    root='./data/cifar/', train=True, download=True, transform=transform_train)
trainset, _ = split_train_val(trainset, val_split=0.1)

valset = torchvision.datasets.CIFAR10(
    root='./data/cifar/', train=True, download=True, transform=transform_test)
_, valset = split_train_val(valset, val_split=0.1)

testset = torchvision.datasets.CIFAR10(
    root='./data/cifar/', train=False, download=True, transform=transform_test)

d_input = 3 if not args.grayscale else 1
d_output = 10

# Dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# Model
print('==> Building model..')
model = SeqBoatModel(
    d_input=d_input,
    d_output=d_output,
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=args.prenorm,
    init_temp_scale=args.init_temp_scale,
)

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

def setup_optimizer(model, lr, weight_decay, epochs):

    # All parameters in the model
    all_parameters = list(model.parameters())

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(all_parameters, betas=(0.9, 0.98), lr=lr, weight_decay=weight_decay)

    
    # Create a lr scheduler
    args.total_num_update = len(trainloader)*args.epochs
    args.warmup_updates = len(trainloader)*int(args.epochs*0.05)
    scheduler = LinearDecaySchedule(args, optimizer)
    return optimizer, scheduler

criterion = nn.CrossEntropyLoss()
optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
)

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
def train(global_step):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        global_step += 1
        scheduler.step_update(global_step)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | Train Acc: %.3f%% (%d/%d)' %
            (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total)
        )


def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Eval Acc: %.3f%% (%d/%d)' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1), 100.*correct/total, correct, total)
            )

    # Save checkpoint.
    if checkpoint:
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt098_Ndp01_base_fn.pth')
            best_acc = acc

        return acc

pbar = tqdm(range(start_epoch, args.epochs))
global_step = 0
for epoch in pbar:
    if epoch == 0:
        pbar.set_description('Epoch: %d' % (epoch))
    else:
        pbar.set_description('Epoch: %d | Val acc: %1.3f' % (epoch, val_acc))
    train(global_step)
    val_acc = eval(epoch, valloader, checkpoint=True)
    eval(epoch, testloader)
    #scheduler.step()