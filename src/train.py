"""
CS289A INTRODUCTION TO MACHINE LEARNING
MACHINE LEARNING FOR LIP READING
Authors: Alejandro Molina & Miguel Fernandez Montes

Script for training and evaluating model
"""

import os
import time
import torch
from src.model import Encoder, Decoder, Attention, LipReading, LipReadingWords

CHECKPOINTS_DIR = 'checkpoints'
num_batches = 2  # TODO get this from somewhere else


def train(model: torch.nn.Module,
          data_loader,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          device: torch.device):
    model.train()  # not sure if necessary
    epoch_loss = 0

    for batch_id, data in enumerate(data_loader):
        inputs, targets = data[0].to(device), data[1].to(device)
        targets = targets.argmax(1)
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / num_batches


def evaluate(model: torch.nn.Module,
             data_loader,
             criterion: torch.nn.Module,
             device: torch.device):
    model.eval()  # not sure if necessary
    epoch_loss = 0

    with torch.no_grad():
        for batch_id, data in enumerate(data_loader):
            inputs, targets = data[0].to(device), data[1].to(device)
            targets = targets.argmax(1)
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            epoch_loss += loss.item()

    return epoch_loss / num_batches


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def run(n_epochs: int,
        train_loader,
        val_loader,
        device: torch.device
        ):
    # for simple word model
    # TODO implement data loading and preprocessing

    print('Initializing model...')

    model = LipReadingWords().to(device)  # might change?
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # might change?
    criterion = torch.nn.CrossEntropyLoss()

    print('\nTraining...')

    # this code is adapted from PyTorch Seq2Seq tutorial
    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f} ')
        print(f'\tVal. Loss: {val_loss:.4f}')

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss},
                   os.path.join(os.pardir, CHECKPOINTS_DIR, f'model_LRW_{epoch}.tar'))


# try with data_loader as a simple list

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f'Using device: {device}')

toy_train_loader = ((torch.randn(64, 2, 5, 120, 120), torch.rand([64, 500])) for i in range(2))
toy_val_loader = ((torch.randn(2, 2, 5, 120, 120), torch.rand([2, 500])) for i in range(2))
run(n_epochs=1, train_loader=toy_train_loader, val_loader=toy_val_loader, device=device)

# TODO weight init
# TODO optimizer config
