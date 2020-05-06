"""
CS289A INTRODUCTION TO MACHINE LEARNING
MACHINE LEARNING FOR LIP READING
Authors: Alejandro Molina & Miguel Fernandez Montes

Script for training and evaluating model
"""

# TODO change num_batches
# TODO lr scheduler
# TODO weight init


import os
import time
import torch
from model import LipReadingWords
# from preprocessing import train_loader, val_loader
from preprocessing_local import train_loader, val_loader


N_EPOCHS = 1
CHECKPOINTS_DIR = 'checkpoints'
TRAIN_SIZE = 0.25 * 500000
# BATCH_SIZE = 64
BATCH_SIZE = 8
# NUM_ITER = TRAIN_SIZE // BATCH_SIZE
NUM_ITER = 20


def train(model: torch.nn.Module,
          data_loader,
          num_iterations: int,
          batch_size: int,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          device: torch.device):
    model.train()
    epoch_loss = 0

    for batch_id, data in enumerate(data_loader(num_iterations, batch_size)):
        inputs, targets = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        print(f'Loss on batch: {loss}')
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / num_iterations


def evaluate(model: torch.nn.Module,
             data_loader,
             num_iterations: int,
             batch_size: int,
             criterion: torch.nn.Module,
             device: torch.device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch_id, data in enumerate(data_loader(num_iterations, batch_size)):
            inputs, targets = data[0].to(device), data[1].to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            epoch_loss += loss.item()

    return epoch_loss / num_iterations


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def run(n_epochs: int,
        train_loader,
        val_loader,
        num_iterations,
        batch_size,
        device: torch.device):
    print('Initializing model...')

    model = LipReadingWords().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, num_iterations, batch_size, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, num_iterations, batch_size, criterion, device)
        val_loss = 0

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


# try with data_loader as a simple generator of random Tensors

# toy_train_loader = ((torch.randn(2, 2, 5, 120, 120), torch.Tensor([12, 50])) for i in range(3))
# toy_val_loader = ((torch.randn(2, 2, 5, 120, 120), torch.Tensor([14, 65])) for i in range(3))
# run(n_epochs=1,
#     train_loader=toy_train_loader,
#     val_loader=toy_val_loader,
#     num_iterations=3,
#     batch_size=2,
#     device=device)


def main():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f'Using device: {device}')

    run(n_epochs=N_EPOCHS,
        train_loader=train_loader,
        val_loader=val_loader,
        num_iterations=NUM_ITER,

        batch_size=BATCH_SIZE,
        device=device)


if __name__ == '__main__':
    main()
