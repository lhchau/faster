import torch
import os
import numpy as np
from .utils import *
from .bypass_bn import *
import torch.nn.functional as F

def loop_one_epoch(
    dataloader,
    net,
    criterion,
    optimizer,
    device,
    logging_dict,
    epoch,
    loop_type='train',
    logging_name=None,
    best_acc=0,
    gradient_accumulation_steps=1
    ):
    loss = 0
    total = 0
    correct = 0
    if loop_type == 'train': 
        net.train()
        for batch_idx, batch in enumerate(dataloader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            opt_name = type(optimizer).__name__
            if opt_name == 'SGD':
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss /= gradient_accumulation_steps
                first_loss.backward()
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    optimizer.step()
                    optimizer.zero_grad()
            elif opt_name == 'SODA':
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss /= gradient_accumulation_steps
                first_loss.backward()
                if (batch_idx + 1) % (gradient_accumulation_steps - 1) == 0 and (batch_idx + 1) != len(dataloader):
                    optimizer.unperturb()
                    optimizer.perturb()
                elif (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    optimizer.unperturb(flush=True)
                    optimizer.step()
                    optimizer.zero_grad()
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                with torch.no_grad():
                    loss += first_loss.item()
                    loss_mean = loss/(batch_idx+1)
                    _, predicted = outputs.max(1)
                    
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    acc = 100.*correct/total
                    
                    progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total))
    else:
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)

                loss += first_loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss_mean = loss/(batch_idx+1)
                acc = 100.*correct/total
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total))
            if acc > best_acc:
                print('Saving best checkpoint ...')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'loss': loss,
                    'epoch': epoch
                }
                save_path = os.path.join('checkpoint', logging_name)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                torch.save(state, os.path.join(save_path, 'ckpt_best.pth'))
                best_acc = acc
            logging_dict[f'{loop_type.title()}/best_acc'] = best_acc
        logging_dict[f'{loop_type.title()}/gen_gap'] = logging_dict['Train/acc'] - acc
                
    logging_dict[f'{loop_type.title()}/loss'] = loss_mean
    logging_dict[f'{loop_type.title()}/acc'] = acc

    if loop_type == 'test': 
        return best_acc, acc