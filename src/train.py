"""
CS289A INTRODUCTION TO MACHINE LEARNING
MACHINE LEARNING FOR LIP READING
Authors: Alejandro Molina & Miguel Fernandez Montes
Script for training and evaluating model



__name__ = "train.py"
__author__ = "Alejandro Molina & Miguel Fernandez Montes "
__version__ = "3.6.0"
__maintainer__ = "Alejandro Molina Sanchez"
__email__ = "miguel_fmontes@berkeley.edu
             alejandro_molina@berkeley.edu"

__date__ = "05/01/2020"


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
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from model import LipReadingWords
from preprocessing import train_loader, val_loader


NUM_EPOCHS = config.NUM_EPOCHS
TRAIN_SIZE = config.TRAIN_SIZE
VAL_SIZE = config.VAL_SIZE
BATCH_SIZE = config.BATCH_SIZE
NUM_ITER_TRAIN = int(TRAIN_SIZE / BATCH_SIZE)
NUM_ITER_VAL = int(VAL_SIZE / BATCH_SIZE)
# NUM_ITER = 2
MODE = config.MODE

CHECKPOINTS_DIR = config.CHECKPOINTS_DIR
LOGS_DIR = config.LOGS_DIR
LOAD_MODEL = config.LOAD_MODEL
LOAD_PATH = config.LOAD_PATH
LOAD_NAME = config.LOAD_NAME
SAVE_PATH = config.SAVE_PATH
SAVE_NAME = config.SAVE_NAME


def topk_accuracy(outputs, targets, k):
    batch_size = targets.size(0)
    _, pred = torch.topk(outputs, k, dim=1)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred)).sum().item()
    return correct


def train(model: torch.nn.Module,
          data_loader,
          epoch: int,
          num_iterations: int,
          batch_size: int,
          optimizer: torch.optim.Optimizer,
          scheduler,
          criterion: torch.nn.Module,
          device: torch.device,
          save_dir,
          save_name,
          writer):
    model.train()
    
    epoch_loss = 0
    running_loss = 0
    total = 0
    correct = 0
    top5_correct = 0

    for batch_id, data in enumerate(data_loader(num_iterations, batch_size)):
        inputs, targets = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        print(f'Train loss on batch: {loss}') # necessary?
        
        loss.backward()

        optimizer.step()
        
        predictions = outputs.argmax(1)
        
        total += targets.size(0)
        correct += (predictions == targets).sum().item()
        
        # top 5 accuracy 
        top5_correct += topk_accuracy(outputs, targets, 5)

        epoch_loss += loss.item()
        running_loss += loss.item()
            
        if (batch_id % 10) == 9:
            writer.add_scalar('Train batch loss',
                              running_loss / 10,
                              epoch * num_iterations + batch_id)
            
            writer.add_scalar('Train batch accuracy',
                              correct * 100 / total,
                              epoch * num_iterations + batch_id)
            
            writer.add_scalar('Train batch top 5 accuracy',
                              top5_correct * 100 / total,
                              epoch * num_iterations + batch_id)
            
            running_loss = 0
            correct = 0
            top5_correct = 0
            total = 0
            
        if (batch_id % 100) == 99:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'batch_loss': loss.item()},
                       os.path.join(save_dir, save_name + f'_train_{epoch * num_iterations + batch_id}.tar'))

    return epoch_loss / num_iterations


def evaluate(model: torch.nn.Module,
             data_loader,
             num_iterations: int,
             batch_size: int,
             criterion: torch.nn.Module,
             device: torch.device):
    model.eval()
    
    epoch_loss = 0
    total = 0
    correct = 0
    top5_correct = 0

    with torch.no_grad():
        for batch_id, data in enumerate(data_loader(num_iterations, batch_size)):
            inputs, targets = data[0].to(device), data[1].to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            print(f'Val. loss on batch: {loss}')
            
            predictions = outputs.argmax(1)

            epoch_loss += loss.item()
            
            total += targets.size(0)
            correct += (predictions == targets).sum().item()

            # top 5 accuracy 
            top5_correct += topk_accuracy(outputs, targets, 5)


    return epoch_loss / num_iterations, correct * 100 / total, top5_correct * 100 / total


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
        num_iter_train: int,
        num_iter_val: int,
        batch_size: int,
        device: torch.device,
        mode,
        reload_model,
        reload_dir,
        reload_file,
        save_dir,
        save_name,
        log_dir):
    print('Initializing model...')

    model = LipReadingWords().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    writer = SummaryWriter(log_dir)
    
    if reload_model:
        try:
            checkpoint = torch.load(os.path.join(reload_dir, reload_file))
            print(f'Reloading {reload_file}')
        except:
            checkpoint = torch.load(get_last_file(reload_dir))
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        print(f'last epoch {last_epoch}')
        print('Reloaded model!')
        
    else:
        last_epoch = -1
    
    # scheduler = MultiStepLR(optimizer, milestones=[2,4,6],
    #                         gamma=0.1, last_epoch=last_epoch)
    
    # CHANGE - for epoch
    scheduler = None
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.0000001 # changed on May 09th 04:15
    # CHANGE
    
    initial_epoch = last_epoch + 1
    criterion = torch.nn.CrossEntropyLoss()
    
    if mode == 'train':
        for epoch in range(initial_epoch, initial_epoch + n_epochs):
            start_time = time.time()

            train_loss = train(model, train_loader, epoch, num_iter_train, batch_size,
                               optimizer, scheduler, criterion, device, save_dir, save_name, writer)
            val_loss, val_acc, val_top5acc = evaluate(model, val_loader, num_iter_val, batch_size,
                                                      criterion, device)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.4f} ')
            print(f'\tVal. Loss: {val_loss:.4f}')

            writer.add_scalar('Train epoch loss', train_loss, epoch)
            
            writer.add_scalar('Val. epoch loss', val_loss, epoch)
            writer.add_scalar('Val. epoch acc', val_acc, epoch)
            writer.add_scalar('Val. epoch top-5 acc', val_top5acc, epoch)
            
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss},
                       os.path.join(save_dir, save_name + f'_{epoch}.tar'))
    
    elif mode == 'test':
        val_loss, val_acc, val_top5acc = evaluate(model, val_loader, num_iter_val, batch_size,
                                                  criterion, device)
        
        writer.add_scalar('Val. epoch loss', val_loss, last_epoch)
        writer.add_scalar('Val. epoch acc', val_acc, last_epoch)
        writer.add_scalar('Val. epoch top-5 acc', val_top5acc, last_epoch)
        
        print(f'\tVal. Loss: {val_loss:.4f}')
        print(f'\tVal. Acc: {val_acc:.4f}')
        print(f'\tVal. top-5 Acc: {val_top5acc:.4f}')
    
    else:
        raise NotImplementedError


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
        
    # A PINCHO
    # model_name = 'model_LRW_proof'
    # reload_dir = CHECKPOINTS_DIR
    # reload_file = 'model_LRW_curr_0.tar'
    # reload = False
    # save_dir = CHECKPOINTS_DIR
    # log_dir = '/home/alemosan/lipReading/logs/proof'
    # A PINCHO
    
    run(n_epochs=NUM_EPOCHS,
        train_loader=train_loader,
        val_loader=val_loader,
        num_iter_train=NUM_ITER_TRAIN,
        num_iter_val=NUM_ITER_VAL,
        batch_size=BATCH_SIZE,
        device=device,
        mode=MODE,
        reload_model=LOAD_MODEL,
        reload_dir=LOAD_PATH,
        reload_file=LOAD_NAME,
        save_dir=SAVE_PATH,
        save_name=SAVE_NAME,
        log_dir=LOGS_DIR
       )


if __name__ == '__main__':
    main()
