#!/home/sci/ricbl/Documents/virtualenvs/dgx_python2_pytorch0.1/bin/python
#SBATCH --time=0-30:00:00 # walltime, abbreviated by -t
#SBATCH --nodes=1 # number of cluster nodes, abbreviated by -N
#SBATCH --mincpus=8
#SBATCH --gres=gpu:1
#SBATCH -o dgx_log/slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e dgx_log/slurm-%j.err-%N # name of the stderr, using the job number (%j) and the first node (%N)
#SBATCH --mem=50G

import torch
import sys
import os
import time

sys.path.append(os.getcwd())
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import logging
timestamp = time.strftime("%Y%m%d-%H%M%S")
logging.basicConfig( filename = 'log/log'+timestamp+'.txt', level=logging.INFO)
print('log'+timestamp+'.txt')
import numpy as np
np.set_printoptions(threshold=np.inf)
import torch.nn as nn
from torch import optim
import torch.backends.cudnn as cudnn
import torchvision
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from random import *
import pandas as pd
from torch.utils.data import DataLoader

import model_loading
import utils
from configs import configs
import outputs
import inputs

logging.info('Using PyTorch ' +str(torch.__version__))

NUM_WORKERS = 1 
# faster to have one worker and use non thread safe h5py to load preprocessed images
# than other tested options

labels_columns = configs.get_enum_return('get_labels_columns')
configs.load_predefined_set_of_configs('densenet') #'densenet', 'frozen_densenet'

def run_epoch(loader, model, criterion, optimizer, epoch=0, n_epochs=0, train=True, ranges_labels = None):
    # modified from https://github.com/gpleiss/temperature_scaling
    time_meter = utils.Meter(name='Time', cum=True)
    loss_meter = utils.Meter(name='Loss', cum=False)
    if train:
        model.train()
    else:
        model.eval()

    end = time.time()
    lossMean = 0.0 
    lossValNorm = 0.0
    y_corr = np.empty([0,len(labels_columns)])
    y_pred = np.empty([0,len(labels_columns)])
    for i, (input, target) in enumerate(loader):
        if train:
            optimizer.zero_grad()

        # Forward pass
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        input_var = torch.autograd.Variable(input, volatile=(not train))
        target_var = torch.autograd.Variable(target, volatile=(not train))

        output_var = model(input_var)
        loss = criterion(output_var, target_var)
        
        output_var_fixed = output_var.data.cpu().numpy()
        target_var_fixed = target_var.data.cpu().numpy()
        
        if configs['use_log_transformation']:
            output_var_fixed = np.exp(output_var_fixed)
            target_var_fixed = np.exp(target_var_fixed)
        if configs['network_output_kind']=='sigmoid':
            output_var_fixed = sigmoid_denormalization(output_var_fixed, ranges_labels)
            target_var_fixed = sigmoid_denormalization(target_var_fixed, ranges_labels)
            
        y_corr = np.concatenate((y_corr, target_var_fixed), axis = 0)
        y_pred = np.concatenate((y_pred, output_var_fixed), axis = 0)
        
        # Backward pass
        if train:
            loss.backward()
            optimizer.step()
            optimizer.n_iters = optimizer.n_iters + 1 if hasattr(optimizer, 'n_iters') else 1
        
        batch_time = time.time() - end
        end = time.time()

        # Log errors
        time_meter.update(batch_time)
        loss_meter.update(loss)
        if not train:
            pass
        #lossMean += len(input)*loss.data.cpu().numpy()
        lossMean += len(input)*loss.data[0]
        lossValNorm += len(input)
    
    return time_meter.value(), loss_meter.value(), np.atleast_1d(lossMean/lossValNorm), y_corr, y_pred

def sigmoid_denormalization(np_array, ranges_labels):
    for i in np_array.shape[0]:
        col_name = labels_columns[i]
        np_array[i,:] =  np_array[i,:]*ranges_labels[col_name][1]*safety_constant[col_name][1]+ranges_labels[col_name][0]*safety_constant[col_name][0]
    return np_array

def count_unique_images_and_pairs(images_to_use, all_examples):
    images_to_use['old_index_pa'] = images_to_use.index
    lat_images = all_examples[all_examples['position'].isin(['LAT'])]
    lat_images['old_index_lat'] = lat_images.index
    b = pd.merge(images_to_use, lat_images, on=['subjectid', 'crstudy'])
    print(len(images_to_use['image_index'].unique()))
    print(len(images_to_use['old_index_pa'].unique()))
    print(len(b['image_index_x'].unique()))
    print(len(b['image_index_y'].unique()))
    print(len(b['old_index_pa'].unique()))
    print(len(b['old_index_lat'].unique()))

# creates models, optimizers, criterion and dataloaders for all models that will be trained in parallel
def load_training_pipeline(images_to_use, transformSequence, num_ftrs):
  
    models = []
    criterions = []
    optimizers = []
    schedulers = []
    train_loaders = []
    test_loaders = []
    
    configs.add_enum_return('training_pipeline', 'one_vs_all','get_training_range', lambda: images_to_use['subjectid'].unique())
    configs.add_enum_return('training_pipeline', 'simple','get_training_range', lambda: range(1))
    configs.add_enum_return('training_pipeline', 'ensemble','get_training_range', lambda: range(configs['total_ensemble_models']))


    training_range = configs.get_enum_return('get_training_range')
    
    for i in training_range:
        if configs['training_pipeline']=='one_vs_all':
            train_images = images_to_use[images_to_use['subjectid']!=i]  
            test_images = images_to_use[images_to_use['subjectid']==i]
        else:
            subjectids = np.array(images_to_use['subjectid'].unique())
            if configs['use_fixed_test_set']:
                prng = np.random.RandomState(83936018)
            else:
                prng = np.random.RandomState()
            queue = prng.permutation(subjectids.shape[0])
            testids = set(subjectids[queue[:int(0.2*subjectids.shape[0])]])
            test_images = images_to_use.loc[images_to_use['subjectid'].isin(testids)]
            train_images = images_to_use.loc[~images_to_use['subjectid'].isin(testids)]  
            assert(len(pd.merge(test_images, train_images, on=['subjectid'])) == 0)
        

        logging.info('size training: '+str(np.array(train_images).shape[0]))
        logging.info('size test: '+str(np.array(test_images).shape[0]))
        
        train_dataset = inputs.DatasetGenerator(train_images, transformSequence)
        
        train_indices = torch.randperm(len(train_dataset))  
        train_loader = DataLoader(dataset=train_dataset, batch_size=configs['BATCH_SIZE'],
                                sampler=SubsetRandomSampler(train_indices), num_workers=NUM_WORKERS, pin_memory=True, drop_last = True)                          
        
        test_dataset = inputs.DatasetGenerator(test_images, transformSequence)
        test_loader = DataLoader(dataset=test_dataset, batch_size=configs['BATCH_SIZE'],
                                shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        logging.info('finished loaders and generators. starting models')
        
        model = model_loading.get_model(num_ftrs)
        
        logging.info('finished models. starting criterion')
        
        criterion = configs.get_enum_return('get_loss')

        optimizer = optim.Adam( model.parameters(), lr=configs['initial_lr'] , weight_decay=configs['l2_reg'])
        scheduler = utils.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min', verbose = True)
        models = models + [model]
        criterions = criterions + [criterion]
        optimizers = optimizers + [optimizer]
        schedulers = schedulers + [scheduler]
        train_loaders = train_loaders + [train_loader]
        test_loaders = test_loaders + [test_loader]
    return models, criterions, optimizers, schedulers, train_loaders, test_loaders
    
def train(models, criterions, optimizers, schedulers, train_loaders, test_loaders, ranges_labels):    # Train model
    n_epochs = configs['N_EPOCHS']
    logging.info('starting training')
    
    training_range = configs.get_enum_return('get_training_range')
  
    for epoch in range(1, n_epochs + 1):    

        all_val_losses =np.array([])
        all_train_losses =np.array([])
        #select patient id instead of line
        for i in training_range:
            
            model = models[i]
            criterion = criterions[i]
            optimizer = optimizers[i]
            train_loader = train_loaders[i]
            test_loader = test_loaders[i]
        
            _, _, loss, y_corr_train, y_pred_train = run_epoch(
                loader=train_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                epoch=epoch,
                n_epochs=n_epochs,
                train=True,
                ranges_labels = ranges_labels
                ) 
            
            all_train_losses = np.concatenate((all_train_losses, loss), axis=0)
            _, _, loss, y_corr_test, y_pred_test = run_epoch(loader=test_loader,
                model=model,
                criterion=criterion,
                optimizer=None,
                epoch=epoch,
                n_epochs=n_epochs,
                train=False,
                ranges_labels = ranges_labels)
            if configs['use_lr_scheduler']:
                schedulers[i].step(loss)
            all_val_losses = np.concatenate((all_val_losses, loss), axis=0)
            
            if configs['training_pipeline']=='simple':
                ys_corr_train = y_corr_train
                ys_pred_train = y_pred_train
                ys_corr_test = y_corr_test
                ys_pred_test = y_pred_test
            elif configs['training_pipeline']=='one_vs_all':
                ys_corr_train = np.concatenate((ys_corr_train, y_corr_train), axis = 0)
                ys_pred_train = np.concatenate((ys_pred_train, y_pred_train), axis = 0)
                ys_corr_test = np.concatenate((ys_corr_test, y_corr_test), axis = 0)
                ys_pred_test = np.concatenate((ys_pred_test, y_pred_test), axis = 0)
            elif configs['training_pipeline']=='ensemble':
                ys_corr_train = np.concatenate((ys_corr_train, y_corr_train), axis = 1)
                ys_pred_train = np.concatenate((ys_pred_train, y_pred_train), axis = 1)
                ys_corr_test = np.concatenate((ys_corr_test, y_corr_test), axis = 1)
                ys_pred_test = np.concatenate((ys_pred_test, y_pred_test), axis = 1)
                
        if epoch==n_epochs:
            if configs['training_pipeline']=='ensemble':
                ys_corr_train = np.mean(ys_corr_train, axis = 1)
                ys_pred_train = np.mean(ys_pred_train, axis = 1)
                ys_corr_test = np.mean(ys_corr_test, axis = 1)
                ys_pred_test = np.mean(ys_pred_test, axis = 1)
                
            outputs.report_final_results(ys_corr_train , ys_pred_train, train = True)
            outputs.report_final_results(ys_corr_test , ys_pred_test, train = False)
        
        logging.info('Train loss ' + str(epoch) + '/' + str(n_epochs) + ': ' +str(np.average(all_train_losses)))
        logging.info('Val loss ' + str(epoch) + '/' + str(n_epochs) + ': '+str(np.average(all_val_losses)))

def main():
    cudnn.benchmark = False
    configs.log_configs()
    
    logging.info('started feature loading')
    
    all_images, transformSequence, num_ftrs = inputs.get_images()
    
    logging.info('ended feature loading. started label loading')
    
    all_labels, ranges_labels = inputs.get_labels()
    
    logging.info('ended label loading')
    
    images_to_use = pd.merge(all_images, all_labels, on=['subjectid', 'crstudy'])
    
    models, criterions, optimizers, schedulers, train_loaders, test_loaders = load_training_pipeline(images_to_use, transformSequence, num_ftrs)
    train(models, criterions, optimizers, schedulers, train_loaders, test_loaders, ranges_labels)
    
if __name__ == '__main__':
    main()