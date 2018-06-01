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


configs.load_predefined_set_of_configs('fc20180524') #'densenet', 'frozen_densenet', 'p1', 'fc20180524'
configs['save_model']= True
configs['N_EPOCHS'] = 50
configs['BATCH_SIZE'] = 1
'''
#configs['training_pipeline']= 'ensemble'
configs['use_true_predicted']= True
#configs['pre_transformation'] = 'boxcox'
configs['N_EPOCHS'] = 50
configs['use_lateral']= True
#configs['kind_of_loss']='l1'
configs['kind_of_loss']='relative_mse'
configs['network_output_kind']='softplus'
#configs['labels_to_use'] ='two_ratios'
configs['labels_to_use'] ='two_predrug_absolute'
configs['trainable_densenet'] = True
#configs.load_predefined_set_of_configs('densenet') 
#configs['output_copd'] =False
configs['use_extra_inputs']= True
configs['use_batchnormalization_hidden_layers']= True
configs['use_random_crops'] = True
configs['positions_to_use'] = ['PA', 'AP']
configs['dropout_batch_normalization_last_layer']=True
configs['densenet_dropout']=0.25
configs['l2_reg_fc'] = 0.0
configs['initial_lr_cnn'] = 1e-04
configs['use_dropout_hidden_layers'] = 0.25
configs['initial_lr_fc'] = 0.0001
configs['BATCH_SIZE'] = 22
#configs['use_copd_definition_as_label'] = True
#configs['output_copd']= True
'''


configs.open_get_block_set()

logging.basicConfig( filename = 'log/log'+configs['timestamp']+'.txt', level=logging.INFO)

print('log'+configs['timestamp']+'.txt')

labels_columns = configs['get_labels_columns']

import model_loading
import utils
import metrics
import inputs
import time
#import visualization

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

def run_epoch(loaders, model, criterion, optimizer, epoch=0, n_epochs=0, train=True):
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
    y_corr, y_pred, y_corr_all = (np.empty([0,x]) for x in (len(labels_columns),len(labels_columns),len(configs['all_output_columns'])))

    for i, (input, target, column_values, extra_inputs, filepath) in enumerate(loaders):
        if train:
            optimizer.zero_grad()
        # Forward pass
        input_var, extra_inputs_var, target_var = ((torch.autograd.Variable(var.cuda(async=True, device = 0), volatile=(not train)) 
                                               if (not isinstance(var,(list,))) else None) for var in (input,extra_inputs,target))
        
        output_var = model(input_var, extra_inputs_var)
        
        losses = []
        
        for k in range(target.size()[1]):
            losses.append(criterion[k]['weight']*criterion[k]['loss'](output_var[:,k], target_var[:,k]))
        loss = sum(losses)
        
        #output_var_fixed, target_var_fixed = (x.data.cpu().numpy() for x in (output_var, target_var))
        output_var_fixed = output_var.data.cpu().numpy()
        target_var_fixed = target.numpy()
        all_target_var_fixed = column_values.numpy()
        
        y_corr, y_pred, y_corr_all = (np.concatenate(x, axis = 0) for x in ((y_corr, target_var_fixed), (y_pred, output_var_fixed), (y_corr_all, all_target_var_fixed)))
        
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
    return time_meter.value(), loss_meter.value(), np.atleast_1d(lossMean/lossValNorm), y_corr, y_pred, y_corr_all

def select_columns_one_table_after_merge(df, suffix, keep_columns=[]):
    to_select = [x for x in df if x.endswith(suffix)]+keep_columns
    to_return = df[to_select]
    to_return.columns = [(x[:-(len(suffix))] if (x not in keep_columns) else x) for x in to_return ]
    return to_return

def get_loader(set_of_images, cases_to_use, all_labels, transformSequence, train, verbose = True):
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
        
        if verbose:
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
def load_training_pipeline(cases_to_use, all_images, all_labels, trainTransformSequence, testTransformSequence, num_ftrs):
  
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
        
        train_loader=get_loader(all_images, train_images, all_labels, trainTransformSequence, train = True)
        if i == 0:
            eval_train_loader=get_loader(all_images, train_images, all_labels, testTransformSequence, train = False, verbose = False)
        test_loader=get_loader(all_images, test_images, all_labels, testTransformSequence, train= False)

        logging.info('finished loaders and generators. starting models')
        
        model = model_loading.get_model(num_ftrs)
        
        logging.info('finished models. starting criterion')
        
        def relative_error_mse_loss(input, target):
            return torch.sum(torch.mean((torch.max(input,target)/torch.min(input,target)-1) ** 1, dim = 0))
        
        #defining what loss function should be used
        losses_dict = {'l1':nn.L1Loss(size_average = True).cuda(), 
          'l2':nn.MSELoss(size_average = True).cuda(), 
          'smoothl1':nn.SmoothL1Loss(size_average = True).cuda(), 
          'bce':nn.BCELoss(size_average = True).cuda(),
          'relative_mse':relative_error_mse_loss}
        criterion = [{'loss':losses_dict[configs['get_individual_kind_of_loss'][k]], 'weight':configs['get_individual_loss_weights'][k]} for k in configs['get_labels_columns']]

        '''
        optimizer = optim.Adam( [
          {'params':model.module.fc.parameters(), 'lr':configs['initial_lr_fc'], 'weight_decay':configs['l2_reg_fc']}, 
          {'params':model.module.cnn.parameters(), 'lr':configs['initial_lr_cnn'], 'weight_decay':configs['l2_reg_cnn']}
          ], lr=configs['initial_lr_fc'] , weight_decay=configs['l2_reg_fc'])
        '''
        optimizer = optim.Adam( [
          {'params':model.module.final_layers.parameters(), 'lr':configs['initial_lr_fc'], 'weight_decay':configs['l2_reg_fc']}, 
          {'params':model.module.cnn.parameters(), 'lr':configs['initial_lr_cnn'], 'weight_decay':configs['l2_reg_cnn']}
          ], lr=configs['initial_lr_fc'] , weight_decay=configs['l2_reg_fc'])
                
        scheduler = utils.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min', verbose = True)
        models = models + [model]
        criterions = criterions + [criterion]
        optimizers = optimizers + [optimizer]
        schedulers = schedulers + [scheduler]
        train_loaders = train_loaders + [train_loader]
        test_loaders = test_loaders + [test_loader]
    return models, criterions, optimizers, schedulers, train_loaders, test_loaders, eval_train_loader

def concat_with_expansion(x,y, axis):
    return np.concatenate((x, np.expand_dims(y, axis = 2)), axis = axis)

def save_model(models):
    for i, model in enumerate(models):
        torch.save(model.state_dict(), './models/' + configs['output_model_name'] + '_' + str(i))

def load_model(models, input_model_name):
    for i, model in enumerate(models):
        model.load_state_dict(torch.load('./models/' + input_model_name + '_' + str(i)))
    
def train(models, criterions, optimizers, schedulers, train_loaders, test_loaders, eval_train_loader):    # Train model
    n_epochs = configs['N_EPOCHS']
    logging.info('starting training')
    
    training_range = configs['get_training_range']
    
    train_dataset_size, test_dataset_size  = (np.sum([len(x[0]) for x in k[0]]) for k in ([eval_train_loader], test_loaders))
    ys_corr_train, ys_pred_train, ys_corr_all_train, ys_corr_test, ys_pred_test, ys_corr_all_test = (np.empty([x,y,0]) for x, y in \
      ((train_dataset_size, len(labels_columns)),(train_dataset_size, len(labels_columns)),(train_dataset_size, len(configs['all_output_columns'])), \
        (test_dataset_size, len(labels_columns)),(test_dataset_size, len(labels_columns)), (test_dataset_size, len(configs['all_output_columns']))))
    
    if configs['training_pipeline']=='simple':
        concatenate_funcation = lambda x,y: y
    elif configs['training_pipeline']=='one_vs_all':
        concatenate_funcation = lambda x,y: concat_with_expansion(x,y, 0)
    elif configs['training_pipeline']=='ensemble':
        concatenate_funcation = lambda x,y: concat_with_expansion(x,y, 2)
                
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
        
            _, _, loss, _, _, _ = run_epoch(
                loaders=train_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                epoch=epoch,
                n_epochs=n_epochs,
                train=True
                ) 
            
            all_train_losses = np.concatenate((all_train_losses, loss), axis=0)
            _, _, loss, y_corr_test, y_pred_test, y_corr_all_test = run_epoch(
                loaders=test_loader,
                model=model,
                criterion=criterion,
                optimizer=None,
                epoch=epoch,
                n_epochs=n_epochs,
                train=False
                )
            
            if epoch==n_epochs:
                _, _, _, y_corr_train, y_pred_train, y_corr_all_train = run_epoch(
                      loaders=eval_train_loader,
                      model=model,
                      criterion=criterion,
                      optimizer=None,
                      epoch=epoch,
                      n_epochs=n_epochs,
                      train=False
                    )
                
                ys_corr_train, ys_pred_train, ys_corr_all_train, ys_corr_test, ys_pred_test, ys_corr_all_test = \
                  (concatenate_funcation(x,y) for x,y in \
                    ((ys_corr_train,y_corr_train),(ys_pred_train,y_pred_train),(ys_corr_all_train,y_corr_all_train), (ys_corr_test,y_corr_test),(ys_pred_test,y_pred_test), (ys_corr_all_test,y_corr_all_test)))
        
            if configs['use_lr_scheduler']:
                schedulers[i].step(loss)
            all_val_losses = np.concatenate((all_val_losses, loss), axis=0)
            
        logging.info('Train loss ' + str(epoch) + '/' + str(n_epochs) + ': ' +str(np.average(all_train_losses)))
        logging.info('Val loss ' + str(epoch) + '/' + str(n_epochs) + ': '+str(np.average(all_val_losses)))
        
        if epoch==n_epochs:
            save_model(models)
            if configs['training_pipeline']=='ensemble':
                ys_corr_train, ys_pred_train, ys_corr_all_train, ys_corr_test, ys_pred_test, ys_corr_all_test = \
                  ( np.mean(x, axis = 2) for x in (ys_corr_train,ys_pred_train,ys_corr_all_train, ys_corr_test,ys_pred_test, ys_corr_all_test))
            
            metrics.report_final_results(ys_corr_train , ys_pred_train, ys_corr_all_train, train = True)
            metrics.report_final_results(ys_corr_test , ys_pred_test, ys_corr_all_test, train = False)

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
    
    all_images, trainTransformSequence, testTransformSequence, num_ftrs = inputs.get_images()
    
    logging.info('ended feature loading. started label loading')
    
    all_labels = inputs.get_labels()
    
    logging.info('ended label loading')
    cases_to_use = merge_images_and_labels(all_images, all_labels)
    
    models, criterions, optimizers, schedulers, train_loaders, test_loaders, eval_train_loader = load_training_pipeline(cases_to_use, all_images, all_labels, trainTransformSequence, testTransformSequence, num_ftrs)
    
    load_model(models, 'model20180525-162109-1987')
    '''
    for i, model in enumerate(models):
        visualization.GradCAM(model).execute(test_loaders[i])
        1/0
    '''
    train(models, criterions, optimizers, schedulers, train_loaders, test_loaders, eval_train_loader)
    
if __name__ == '__main__':
    main()