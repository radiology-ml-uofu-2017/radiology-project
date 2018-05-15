#!/home/sci/ricbl/Documents/virtualenvs/dgx_python2_pytorch0.1/bin/python
#SBATCH --time=0-30:00:00 # walltime, abbreviated by -t
#SBATCH --nodes=1 # number of cluster nodes, abbreviated by -N
#SBATCH --mincpus=8
#SBATCH --gres=gpu:1
#SBATCH -o dgx_log/slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e dgx_log/slurm-%j.err-%N # name of the stderr, using the job number (%j) and the first node (%N)
#SBATCH --mem=50G
from future.utils import raise_with_traceback
import torch
import sys
import os
import time
import argparse

sys.path.append(os.getcwd())

import logging


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
from itertools import izip
try:
    import cPickle as pickle
except:
    import _pickle as pickle 
    
from configs import configs


configs.load_predefined_set_of_configs('p12') #'densenet', 'frozen_densenet', 'p1'
#configs['labels_to_use'] ='only_absolute'
#configs.load_predefined_set_of_configs('densenet') 
#configs['output_copd'] =False

configs.open_get_block_set()

logging.basicConfig( filename = 'log/log'+configs['timestamp']+'.txt', level=logging.INFO)

print('log'+configs['timestamp']+'.txt')

labels_columns = configs['get_labels_columns']

import model_loading
import utils
import metrics
import inputs
import time

logging.info('Using PyTorch ' +str(torch.__version__))

NUM_WORKERS = 1 
# faster to have one worker and use non thread safe h5py to load preprocessed images
# than other tested options

parser = argparse.ArgumentParser()
parser.add_argument('-c','--cvd', required=False)
args = vars(parser.parse_args())

if configs['machine_to_use'] == 'titan':
    if args['cvd'] is None:
        raise_with_traceback(ValueError('You should set Cuda Visible Devices (-c or --cvd) when using titan'))
    os.environ['CUDA_VISIBLE_DEVICES'] = args['cvd']

def run_epoch(loaders, model, criterion, optimizer, epoch=0, n_epochs=0, train=True, ranges_labels = None):
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
    for i, (input, target) in enumerate(loaders):
        if train:
            optimizer.zero_grad()
        # Forward pass
        
        output_var = model(torch.autograd.Variable(input.cuda(async=True, device = 0), volatile=(not train)))

        target_var = torch.autograd.Variable(target.cuda(async=True, device = 0), volatile=(not train))
        
        losses = []
        for k in range(target.size()[1]):
            losses.append(criterion[k](output_var[:,k], target_var[:,k]))
        loss = sum(losses)
        
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

        this_batch_size = len(input)
        #lossMean += this_batch_size*loss.data.cpu().numpy()
        lossMean += this_batch_size*loss.data[0]
        lossValNorm += this_batch_size
    #logging.info('oi1 ' + str(time.time() - start))
    return time_meter.value(), loss_meter.value(), np.atleast_1d(lossMean/lossValNorm), y_corr, y_pred

def sigmoid_denormalization(np_array, ranges_labels):
    for i in range(np_array.shape[1]):
        col_name = labels_columns[i]
        np_array[:,i] =  np_array[:,i]*ranges_labels[col_name][1]*configs['sigmoid_safety_constant'][col_name][1]+ranges_labels[col_name][0]*configs['sigmoid_safety_constant'][col_name][0]
    return np_array

def select_columns_one_table_after_merge(df, suffix, keep_columns=[]):
    to_select = [x for x in df if x.endswith(suffix)]+keep_columns
    to_return = df[to_select]
    to_return.columns = [(x[:-(len(suffix))] if (x not in keep_columns) else x) for x in to_return ]
    return to_return

def get_loader(set_of_images, cases_to_use, all_labels, transformSequence, train):
    cases_to_use_on_set_of_images = []
    for i in range(len(set_of_images)):
        current_df = set_of_images[i].copy(deep=True)
        current_df.columns = current_df.columns.map(lambda x: ((str(x) + '_'+str(i)) if x not in ['subjectid', 'crstudy'] else str(x)))
        if i == 0:
            all_joined_table =  pd.merge(cases_to_use, current_df, on=['subjectid', 'crstudy'])#, suffixes=('_cases', '_'+str(i)))
        else:
            all_joined_table = pd.merge(all_joined_table, current_df, on=['subjectid', 'crstudy'])#, suffixes=('_'+str(i-1), '_'+str(i)))
    for i in range(len(set_of_images)):
        cases_to_use_on_set_of_images.append(pd.merge(select_columns_one_table_after_merge(all_joined_table, '_'+str(i), ['subjectid', 'crstudy']),all_labels, on=['subjectid', 'crstudy']))
        
        logging.info('size ' + ('train' if train else 'test') + ' ' + str(i) +': '+str(np.array(cases_to_use_on_set_of_images[i]).shape[0]))
    t_dataset = inputs.DatasetGenerator(cases_to_use_on_set_of_images, transformSequence)
    if train:
      t_indices = torch.randperm(len(t_dataset))
      t_loader = DataLoader(dataset=t_dataset, batch_size=configs['BATCH_SIZE'],
                          sampler=SubsetRandomSampler(t_indices), num_workers=NUM_WORKERS, pin_memory=True, drop_last = True)
    else:
        t_loader = DataLoader(dataset=t_dataset, batch_size=configs['BATCH_SIZE'],
                                shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return t_loader

# creates models, optimizers, criterion and dataloaders for all models that will be trained in parallel
def load_training_pipeline(cases_to_use, all_images, all_labels, transformSequence, num_ftrs):
  
    models = []
    criterions = []
    optimizers = []
    schedulers = []
    train_loaders = []
    test_loaders = []
    
    configs.add_self_referenced_variable_from_dict('get_training_range', 'training_pipeline', 
                                          {'one_vs_all':cases_to_use['subjectid'].unique(),
                                           'simple': range(1),
                                           'ensemble': range(configs['total_ensemble_models'])})

    training_range = configs['get_training_range']
    
    for i in training_range:
        if configs['training_pipeline']=='one_vs_all':
            train_images = cases_to_use[cases_to_use['subjectid']!=i]  
            test_images = cases_to_use[cases_to_use['subjectid']==i]
        else:
            subjectids = np.array(cases_to_use['subjectid'].unique())
            if configs['use_fixed_test_set']:
                #prng = np.random.RandomState(83936018)
                #queue = prng.permutation(subjectids.shape[0])
                #testids = set(subjectids[queue[:int(0.2*subjectids.shape[0])]])
                with open('./testsubjectids.pkl') as f:  # Python 3: open(..., 'rb')
                    testids = pickle.load(f)
            else:
                prng = np.random.RandomState()
                queue = prng.permutation(subjectids.shape[0])
                testids = set(subjectids[queue[:int(0.2*subjectids.shape[0])]])
            #with open('./testsubjectids.pkl', 'w') as f:  # Python 3: open(..., 'wb')
            #    pickle.dump(testids, f)

            test_images = cases_to_use.loc[cases_to_use['subjectid'].isin(testids)]
            train_images = cases_to_use.loc[~cases_to_use['subjectid'].isin(testids)]  
            assert(len(pd.merge(test_images, train_images, on=['subjectid'])) == 0)

        logging.info('total images training: '+str(np.array(train_images).shape[0]))
        logging.info('total images test: '+str(np.array(test_images).shape[0]))
        
        train_loader=get_loader(all_images, train_images, all_labels, transformSequence, train = True)
        test_loader=get_loader(all_images, test_images, all_labels, transformSequence, train= False)

        logging.info('finished loaders and generators. starting models')
        
        model = model_loading.get_model(num_ftrs)
        
        logging.info('finished models. starting criterion')
        
        #defining what loss function should be used
        losses_dict = {'l1':nn.L1Loss(size_average = True).cuda(), 
          'l2':nn.MSELoss(size_average = True).cuda(), 
          'smoothl1':nn.SmoothL1Loss(size_average = True).cuda(), 
          'bce':nn.BCELoss(size_average = True).cuda()}

        criterion = [(losses_dict[configs['individual_kind_of_loss'][configs['get_labels_columns'][k]]] if configs['get_labels_columns'][k] in list(configs['individual_kind_of_loss'].keys()) else losses_dict[configs['kind_of_loss']]) for k in range(len(configs['get_labels_columns']))]
        
        '''
        optimizer = optim.Adam( [
          {'params':model.module.fc.parameters(), 'lr':configs['initial_lr_fc'], 'weight_decay':configs['l2_reg_fc']}, 
          {'params':model.module.cnn.parameters(), 'lr':configs['initial_lr_cnn'], 'weight_decay':configs['l2_reg_cnn']}
          ], lr=configs['initial_lr_fc'] , weight_decay=configs['l2_reg_fc'])
        '''
        optimizer = optim.Adam( [
          {'params':model.module.fc.parameters(), 'lr':configs['initial_lr_fc'], 'weight_decay':configs['l2_reg_fc']}, 
          {'params':model.module.cnn.parameters(), 'lr':configs['initial_lr_cnn'], 'weight_decay':configs['l2_reg_cnn']}
          ], lr=configs['initial_lr_fc'] , weight_decay=configs['l2_reg_fc'])
                
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
    
    training_range = configs['get_training_range']
    
    ys_corr_train = np.empty([np.sum([len(x[0]) for x in train_loaders[0]]),len(labels_columns),0])
    ys_pred_train = np.empty([np.sum([len(x[0]) for x in train_loaders[0]]),len(labels_columns),0])
    ys_corr_test = np.empty([np.sum([len(x[0]) for x in test_loaders[0]]),len(labels_columns),0])
    ys_pred_test = np.empty([np.sum([len(x[0]) for x in test_loaders[0]]),len(labels_columns),0])
    
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
                loaders=train_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                epoch=epoch,
                n_epochs=n_epochs,
                train=True,
                ranges_labels = ranges_labels
                ) 
            
            all_train_losses = np.concatenate((all_train_losses, loss), axis=0)
            _, _, loss, y_corr_test, y_pred_test = run_epoch(
                loaders=test_loader,
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
                ys_corr_train = np.concatenate((ys_corr_train, np.expand_dims(y_corr_train, axis = 2)), axis = 0)
                ys_pred_train = np.concatenate((ys_pred_train, np.expand_dims(y_pred_train, axis = 2)), axis = 0)
                ys_corr_test = np.concatenate((ys_corr_test, np.expand_dims(y_corr_test, axis = 2)), axis = 0)
                ys_pred_test = np.concatenate((ys_pred_test, np.expand_dims(y_pred_test, axis = 2)), axis = 0)
            elif configs['training_pipeline']=='ensemble':
                ys_corr_train = np.concatenate((ys_corr_train, np.expand_dims(y_corr_train, axis = 2)), axis = 2)
                ys_pred_train = np.concatenate((ys_pred_train, np.expand_dims(y_pred_train, axis = 2)), axis = 2)
                ys_corr_test = np.concatenate((ys_corr_test, np.expand_dims(y_corr_test, axis = 2)), axis = 2)
                ys_pred_test = np.concatenate((ys_pred_test, np.expand_dims(y_pred_test, axis = 2)), axis = 2)
                
        if epoch==n_epochs:
            if configs['training_pipeline']=='ensemble':
                ys_corr_train = np.mean(ys_corr_train, axis = 1)
                ys_pred_train = np.mean(ys_pred_train, axis = 1)
                ys_corr_test = np.mean(ys_corr_test, axis = 1)
                ys_pred_test = np.mean(ys_pred_test, axis = 1)
                
            metrics.report_final_results(ys_corr_train , ys_pred_train, train = True)
            metrics.report_final_results(ys_corr_test , ys_pred_test, train = False)
            
        logging.info('Train loss ' + str(epoch) + '/' + str(n_epochs) + ': ' +str(np.average(all_train_losses)))
        logging.info('Val loss ' + str(epoch) + '/' + str(n_epochs) + ': '+str(np.average(all_val_losses)))

def merge_images_and_labels(all_images, all_labels):
    for i, image_set in enumerate(all_images):
        if i == 0:
            all_images_merged = image_set[['subjectid', 'crstudy']]
        else:
            all_images_merged = pd.merge(all_images_merged, image_set, on=['subjectid', 'crstudy'])[['subjectid', 'crstudy']]
    cases_to_use = pd.merge(all_images_merged, all_labels, on=['subjectid', 'crstudy'])[['subjectid', 'crstudy']].groupby(['subjectid', 'crstudy']).count().reset_index()
    return cases_to_use

def main():
    cudnn.benchmark = False
    configs.log_configs()
    
    logging.info('started feature loading')
    
    all_images, transformSequence, num_ftrs = inputs.get_images()
    
    logging.info('ended feature loading. started label loading')
    
    all_labels, ranges_labels = inputs.get_labels()
    
    logging.info('ended label loading')
    cases_to_use = merge_images_and_labels(all_images, all_labels)
    
    models, criterions, optimizers, schedulers, train_loaders, test_loaders = load_training_pipeline(cases_to_use, all_images, all_labels, transformSequence, num_ftrs)
    train(models, criterions, optimizers, schedulers, train_loaders, test_loaders, ranges_labels)
    
if __name__ == '__main__':
    main()