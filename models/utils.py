import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.evaluation import calculate_auc, calculate_metrics, expected_calibration_error

from importlib import import_module

def standard_train(opt, network, optimizer, loader, _criterion, wandb):
    """Train the model for one epoch"""
    train_loss, auc, no_iter = 0., 0., 0
    tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
    torch.cuda.empty_cache()
    for i, (images, targets, sensitive_attr, index) in enumerate(loader):
        images, targets, sensitive_attr = images.to(opt['device']), targets.to(opt['device']), sensitive_attr.to(opt['device'])
        optimizer.zero_grad()
        outputs, _ = network(images) # returns the fc, avgpool outputs

        loss = _criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        probabilites = F.sigmoid(outputs).cpu().data.numpy()
        targets_np = targets.cpu().data.numpy()
        auc += calculate_auc(probabilites, targets_np)

        # Extend the list with new targets
        tol_output.extend(probabilites.flatten().tolist())
        tol_target.extend(targets_np.flatten().tolist())
        tol_sensitive.extend(sensitive_attr.cpu().data.numpy().flatten().tolist())
        tol_index.extend(index.numpy().flatten().tolist())

        train_loss += loss.item()
        no_iter += 1
        
        if opt['log_freq'] and (i % opt['log_freq'] == 0):
            wandb.log({'Training loss': train_loss / no_iter, 'Training AUC': auc / no_iter})

        torch.cuda.empty_cache()

    auc = 100 * auc / no_iter
    ece = expected_calibration_error(np.array(tol_output), np.array(tol_target))
    train_loss /= no_iter
    return auc, train_loss, ece


def standard_val(opt, network, loader, _criterion, sens_classes, wandb):
    """Compute model output on validation set"""
    tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
    
    val_loss, auc = 0., 0.
    no_iter = 0
    with torch.no_grad():
        for i, (images, targets, sensitive_attr, index) in enumerate(loader):
            images, targets, sensitive_attr = images.to(opt['device']), targets.to(opt['device']), sensitive_attr.to(
                opt['device'])
            outputs, features = network.forward(images)
            loss = _criterion(outputs, targets)
            try:
                val_loss += loss.item()
            except:
                val_loss += loss.mean().item()
            tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
            tol_target += targets.cpu().data.numpy().tolist()
            tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
            tol_index += index.numpy().tolist()
            
            auc += calculate_auc(F.sigmoid(outputs).cpu().data.numpy(),
                                           targets.cpu().data.numpy())
            
            no_iter += 1
            
            if opt['log_freq'] and (i % opt['log_freq'] == 0):
                wandb.log({'Validation loss': val_loss / no_iter, 'Validation AUC': auc / no_iter})

            torch.cuda.empty_cache()
            
    auc = 100 * auc / no_iter
    val_loss /= no_iter
    log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, sens_classes)
    
    return auc, val_loss, log_dict, pred_df


def standard_test(opt, network, loader, _criterion, wandb):
    """Compute model output on testing set"""
    tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []

    with torch.no_grad():
        for i, (images, targets, sensitive_attr, index) in enumerate(loader):
            images, targets, sensitive_attr = images.to(opt['device']), targets.to(opt['device']), sensitive_attr.to(
                opt['device'])
            outputs, features = network.forward(images)

            tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
            tol_target += targets.cpu().data.numpy().tolist()
            tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
            tol_index += index.numpy().tolist()
            
    return tol_output, tol_target, tol_sensitive, tol_index