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
import sys
import glob
import time
import torch
import config
from model import LipReadingWords
from preprocessing import train_loader, val_loader


N_EPOCHS = 5
CHECKPOINTS_DIR = '/home/alemosan/lipReading/checkpoints/'
TRAIN_SIZE = 0.25 * 500_000
BATCH_SIZE = 8
NUM_ITER = int(TRAIN_SIZE / BATCH_SIZE)


def train(model: torch.nn.Module,
          data_loader,
          epoch: int,
          num_iterations: int,
          batch_size: int,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          device: torch.device):
    model.train()
    epoch_loss = 0

    # TODO pass epoch to loader
    for batch_id, data in enumerate(data_loader(num_iterations, batch_size)):
        inputs, targets = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        print(f'Loss on batch: {loss}')
        
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        
        if (batch_id % 1000) == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'batch_loss': loss.item()},
                       os.path.join(CHECKPOINTS_DIR, f'model_LRW_train_{batch_id}.tar'))

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
            print(f'Val. loss on batch: {loss}')

            epoch_loss += loss.item()

    return epoch_loss / num_iterations


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_last_file(source_dir):
    list_of_files = glob.glob(source_dir + 'model_*.tar') # * means all if need specific format then *.csv
    print(list_of_files)
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def run(n_epochs: int,
        train_loader,
        val_loader,
        num_iterations,
        batch_size,
        device: torch.device,
        mode='train',
        reload_model=False,
        reload_dir=CHECKPOINTS_DIR,
        reload_file=None
       ):
    print('Initializing model...')

    model = LipReadingWords().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    if reload_model:
        try:
            checkpoint = torch.load(os.path.join(reload_dir, reload_file))
        except:
            checkpoint = torch.load(get_last_file(reload_dir))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        print('Reloaded model!')
    else:
        last_epoch = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    if mode == 'train':
        for epoch in range(last_epoch, last_epoch + n_epochs):
            start_time = time.time()

            train_loss = train(model, train_loader, epoch, num_iterations, batch_size,
                               optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, num_iterations, batch_size,
                                criterion, device)

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
                       os.path.join(CHECKPOINTS_DIR, f'model_LRW_{epoch}.tar'))
    
    elif mode == 'test':
        train_loss = evaluate(model, train_loader, 50, batch_size,
                              criterion, device)
        val_loss = evaluate(model, val_loader, 50, batch_size,
                            criterion, device)
        print(f'\tTrain. Loss: {train_loss:.4f}')
        print(f'\tVal. Loss: {val_loss:.4f}')
    
    else:
        raise NotImplementedError


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # TODO parse arguments
    mode = 'test'
    reload = True
    
    run(n_epochs=N_EPOCHS,
        train_loader=train_loader,
        val_loader=val_loader,
        num_iterations=NUM_ITER,
        batch_size=BATCH_SIZE,
        device=device,
        mode=mode,
        reload_model=reload,
        reload_file='model_LRW_0.tar'
       )


if __name__ == '__main__':
    main()
