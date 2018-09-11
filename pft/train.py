#!/home/sci/ricbl/Documents/virtualenvs/dgx_python3_pytorch0.4/bin/python
#SBATCH --time=0-30:00:00 # walltime, abbreviated by -t
#SBATCH --nodes=1 # number of cluster nodes, abbreviated by -N
#SBATCH --mincpus=8
#SBATCH --gres=gpu:1
#SBATCH -o dgx_log/slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e dgx_log/slurm-%j.err-%N # name of the stderr, using the job number (%j) and the first node (%N)
#SBATCH --mem=25G
from future.utils import raise_with_traceback
import torch
import sys
import os
import time
import argparse
import math

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
try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip
try:
    import cPickle as pickle
except:
    import _pickle as pickle 
    
from configs import configs

configs.load_predefined_set_of_configs('cnn20180628') #'densenet', 'frozen_densenet', 'p1', 'fc20180524', 'cnn20180628'

configs.load_predefined_set_of_configs('resnet18')

#configs['use_spatial_transformer_network'] = True
#configs.load_predefined_set_of_configs('meanvar_loss')

#configs['output_copd'] = True
#configs['labels_to_use'] = 'fev1_ratio'
#configs['use_copd_definition_as_label'] = True
#configs['positions_to_use']= ['PA']

#configs['fully_connected_kind'] = 'softmax_gate'

#configs['gate_uniformity_loss_multiplier'] = 0.0

#configs['gate_orthogonal_loss_multiplier'] = 0.0

#configs['mutual_exclusivity_loss_multiplier'] = 1./100
#configs['fully_connected_kind'] = 'dsnm'

#configs['load_image_features_from_file'] = False
configs['data_to_use']  = ['2012-2016', '2017']
#configs['BATCH_SIZE']=20
#configs['densenet_dropout'] = 0.0
#configs['use_dropout_hidden_layers'] = 0.0
configs['pretrain_kind']  = 'imagenet'
configs['maximum_date_diff']  = 2
configs['remove_lung_transplants']  = True

configs['n_hidden_layers'] = 2
configs['channels_hidden_layers'] = 256 # only used if configs['n_hidden_layers']>0
configs['use_dropout_hidden_layers'] = 0.25 # 0 turn off dropout; 0.25 gives seems to give about the same results as l = 0.05 # only used if configs['n_hidden_layers']>0
configs['use_batchnormalization_hidden_layers'] = True
configs['remove_pre_avg_pool'] = False
configs['l2_reg_fc'] = 0.0
configs['l2_reg_cnn'] = 0.0
#configs['initial_lr_fc'] = 0.001
#configs['initial_lr_cnn'] = 0.001

#configs['chexnet_architecture'] =  'resnet'
#configs['chexnet_layers'] = 50
#configs['CKPT_PATH']  = 'model_chestxray14_resnet_50.t7'


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

NUM_WORKERS = 0
# faster to have one worker and use non thread safe h5py to load preprocessed images
# than other tested options

parser = argparse.ArgumentParser()
parser.add_argument('-c','--cvd', required=False)
args = vars(parser.parse_args())

if configs['machine_to_use'] == 'titan':
    if args['cvd'] is None:
        raise_with_traceback(ValueError('You should set Cuda Visible Devices (-c or --cvd) when using titan'))
    os.environ['CUDA_VISIBLE_DEVICES'] = args['cvd']

def meanvar_var_loss_calc(output_var, interm_probs_logits, model):
    bins = model.module.final_layers.final_linear.bins_list
    losses = []
    for k in range(len(interm_probs_logits)):
        size0 = output_var[:,k].size()[0]
        size1 = bins[k].size()[1]
        if utils.compare_versions(torch.__version__, '0.4.0')>=0:
            nonlinearity = nn.Softmax(dim = 1).cuda()
        else:
            nonlinearity = nn.Softmax().cuda()
        losses.append(torch.mean(torch.sum(nonlinearity(interm_probs_logits[k])*(output_var[:,k].unsqueeze(1).expand(size0, size1) - bins[k].expand(size0, size1))**2, dim = 1), dim = 0) )
    return sum(losses) #sum?

def meanvar_average_loss_calc(output_var, target_var):
    losses = []
    for k in range(target_var.size()[1]):
        losses.append(nn.MSELoss(size_average = True).cuda()(output_var[:,k], target_var[:,k]))
    return sum(losses) #sum?
  
def round_with_chosen_spacing(tensor, spacing):
    return torch.round(tensor/float(spacing))*spacing #only working for one non-sero digit spacings

def torch_one_hot(batch,depth):
    if utils.compare_versions(torch.__version__, '0.4.0')>=0:
        ones = torch.eye(depth)
    else:
        ones = torch.sparse.torch.eye(depth)
    return ones.index_select(0,batch)
  
def meanvar_bce_loss_calc(interm_probs_logits, target_var, model, write):
    assert(len(interm_probs_logits)==target_var.size()[1])
    spacing = configs['meanvarloss_discretization_spacing']
    losses = []
    bins = model.module.final_layers.final_linear.bins_list
    for k in range(target_var.size()[1]):
        rounded_targets = round_with_chosen_spacing(target_var[:,k], spacing)
        size0 = rounded_targets.size()[0]
        size1 = bins[k].size()[1]
        int_targets = (torch.round((rounded_targets - torch.min(bins[k]).expand(size0))/spacing)).long()
        one_hot_targets = torch.autograd.Variable(torch_one_hot(int_targets.data.cpu(), size1).cuda(async=True, device = 0), requires_grad=False) 
        #outputs with larger range are impacting more in the loss here, maybe not reduce and then average
        losses.append(nn.CrossEntropyLoss(size_average = True).cuda()(interm_probs_logits[k], int_targets))
        if write:
            np.savetxt('rounded_targetsmean'+str(k)+'.csv', rounded_targets.data.cpu().numpy(), delimiter=',') 
            np.savetxt('int_targetsmean'+str(k)+'.csv', int_targets.data.cpu().numpy(), delimiter=',') 
            np.savetxt('one_hot_targetsmean'+str(k)+'.csv', one_hot_targets.data.cpu().numpy(), delimiter=',') 
    if write:
        print(sum(losses))
    return sum(losses) #sum?

def mutual_exclusivity_loss(ws):
    '''
    #print(ws)
    loss = torch.sum(ws * torch.log(ws), dim = 1) / math.log(1./ws.size(1))
    loss = torch.mean(loss, dim = 0)
    #print loss
    return loss
    '''
    for i in range(ws.size(1)):
        if i > 0:
            if i == ws.size(1)-1:
                others_ws = ws[:,0:i]
            else:
                others_ws = torch.cat((ws[:,0:i], ws[:,i+1:]), dim = 1)
        else:
            others_ws = ws[:,i+1:]
        this_loss = ws[:,i]*torch.prod(1-others_ws, dim = 1)
        if i == 0:
            loss = this_loss
        else:
            loss = loss + this_loss
    loss = -torch.mean(loss, dim = 0)
    return loss

def gate_uniformity_loss(ws):
    a = torch.sum(ws, dim = 0)
    if len(a.size())<2:
        a = a.unsqueeze(0)
    b = torch.sum(a, dim = 1)
    probabilities = a/b.expand(a.size(0),a.size(1))
    #print(probabilities)
    loss = torch.sum(probabilities * torch.log(probabilities), dim = 1) / -math.log(1./ws.size(1))
    
    #print(loss)
    return loss

def orthogonality_loss(vs):
    dot_products = torch.matmul(vs, vs.permute(0,2,1))
    dot_products = torch.abs(dot_products*(1-torch.eye(dot_products.shape[1],dot_products.shape[2]).unsqueeze(0).cuda()))
    return torch.mean(dot_products)
  
def run_epoch(loaders, model, criterion, optimizer, epoch=0, n_epochs=0, train=True):
    # modified from https://github.com/gpleiss/temperature_scaling
    if utils.compare_versions(torch.__version__, '0.4.0')>=0:
        torch.set_grad_enabled(train)
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
        output_var, extra_outs = model(input_var, extra_inputs_var, epoch)
        interm_probs_logits = extra_outs['logits']
        averages = extra_outs['averages']
        ws = extra_outs['ws']
        vs = extra_outs['vs']
        
        if not configs['use_mean_var_loss']:
            losses = []
            
            for k in range(target.size()[1]):
                losses.append(criterion[k]['weight']*criterion[k]['loss'](output_var[:,k], target_var[:,k]))
            loss = sum(losses)#just like the comment above: should it be the average here? I probably need to choose a different learning rate for common and non-common variables between outputs  /float(len(losses))
        else:
            write = (i==0 and epoch==50) and train and False
            meanvar_var_loss = meanvar_var_loss_calc(averages, interm_probs_logits, model)
            meanvar_bce_loss = meanvar_bce_loss_calc(interm_probs_logits, target_var, model, write)
            meanvar_average_loss = meanvar_average_loss_calc(averages, target_var)
            loss = configs['multiplier_constant_meanvar_mean_loss']*meanvar_average_loss + configs['multiplier_constant_meanvar_var_loss']*meanvar_var_loss + configs['multiplier_constant_meanvar_bce_loss']*meanvar_bce_loss

            if write:
                np.savetxt('output_varmean.csv', output_var.data.cpu().numpy(), delimiter=',') 
                for k in range(target.size()[1]):
                    np.savetxt('interm_probs_logitsmean'+str(k)+'.csv', interm_probs_logits[k].data.cpu().numpy(), delimiter=',') 
                np.savetxt('targetmean.csv', target.cpu().numpy(), delimiter=',') 
                bins = model.module.final_layers.final_linear.bins_list
                for k in range(target.size()[1]):
                    print(bins[k])
        output_var_fixed = output_var.data.cpu().numpy()
        target_var_fixed = target.numpy()
        all_target_var_fixed = column_values.numpy()
        '''
        if epoch < 10:
            optimizer = optim.Adam( [
                  {'params':model.module.final_layers.fc_part.fc_11.parameters(), 'lr':0.0, 'weight_decay':0.0},
                  {'params':model.module.final_layers.fc_part.fc_12.parameters(), 'lr':configs['initial_lr_fc'], 'weight_decay':configs['l2_reg_fc']},
                  {'params':model.module.final_layers.fc_part.fc_1.parameters(), 'lr':configs['initial_lr_fc'], 'weight_decay':configs['l2_reg_fc']},
                  {'params':model.module.final_layers.spatial_part.parameters(), 'lr':configs['initial_lr_fc'], 'weight_decay':configs['l2_reg_fc']}, 
                  {'params':model.module.cnn.parameters(), 'lr':configs['initial_lr_cnn'], 'weight_decay':configs['l2_reg_cnn']}
          ], lr=configs['initial_lr_fc'] , weight_decay=configs['l2_reg_fc'])
        '''
        if configs['fully_connected_kind']=='softmax_gate':
            print(mutual_exclusivity_loss(ws))
            print(orthogonality_loss(vs))
            print(loss)
            loss =loss + configs['mutual_exclusivity_loss_multiplier']*mutual_exclusivity_loss(ws) + configs['gate_uniformity_loss_multiplier']*gate_uniformity_loss(ws) +  configs['gate_orthogonal_loss_multiplier']*orthogonality_loss(vs)
        y_corr, y_pred, y_corr_all = (np.concatenate(x, axis = 0) for x in ((y_corr, target_var_fixed), (y_pred, output_var_fixed), (y_corr_all, all_target_var_fixed)))
        
        
        # Backward pass
        if train:
            loss.backward()
            optimizer.step()
            optimizer.n_iters = optimizer.n_iters + 1 if hasattr(optimizer, 'n_iters') else 1
            #print(model.module.stn.localisation.linear_out.bias.grad)
        
        batch_time = time.time() - end
        end = time.time()

        # Log errors
        time_meter.update(batch_time)
        loss_meter.update(loss)
        if not train:
            pass

        this_batch_size = len(input)
        if utils.compare_versions(torch.__version__, '0.4.0')<0:
            lossMean += this_batch_size*loss.data[0] #alternative to this: lossMean += this_batch_size*loss.data.cpu().numpy()
        else:
            lossMean += this_batch_size*loss.item()
        lossValNorm += this_batch_size
    if utils.compare_versions(torch.__version__, '0.4.0')>=0:
        torch.set_grad_enabled(True)
    return time_meter.value(), loss_meter.value(), np.atleast_1d(lossMean/lossValNorm), y_corr, y_pred, y_corr_all

def select_columns_one_table_after_merge(df, suffix, keep_columns=[]):
    to_select = [x for x in df if x.endswith(suffix)]+keep_columns
    to_return = df[to_select]
    to_return.columns = [(x[:-(len(suffix))] if (x not in keep_columns) else x) for x in to_return ]
    return to_return

def get_loader(set_of_images, cases_to_use, all_labels, transformSequence, split, verbose = True):
    cases_to_use_on_set_of_images = []
    for i in range(len(set_of_images)):
        current_df = set_of_images[i].copy(deep=True)
        current_df.columns = current_df.columns.map(lambda x: ((str(x) + '_'+str(i)) if x not in ['subjectid', 'crstudy'] else str(x)))
        if i == 0:
            all_joined_table =  pd.merge(cases_to_use, current_df, on=['subjectid', 'crstudy'])
        else:
            all_joined_table = pd.merge(all_joined_table, current_df, on=['subjectid', 'crstudy'])
    for i in range(len(set_of_images)):
        cases_to_use_on_set_of_images.append(pd.merge(select_columns_one_table_after_merge(all_joined_table, '_'+str(i), ['subjectid', 'crstudy']),all_labels, on=['subjectid', 'crstudy']))
        
        if verbose:
            logging.info('size ' + split + ' ' + str(i) +': '+str(np.array(cases_to_use_on_set_of_images[i]).shape[0]))
    t_dataset = inputs.DatasetGenerator(cases_to_use_on_set_of_images, transformSequence)
    if split == 'train':
      if configs['balance_dataset_by_fvcfev1_predrug']:
          column_to_use = cases_to_use_on_set_of_images[0]['fev1fvc_predrug']
          column_cut = pd.cut(column_to_use, bins=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
          weights = (1/8./(column_to_use.groupby(column_cut).transform('count')/column_to_use.count()))

          sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=True)
      else:
          t_indices = torch.randperm(len(t_dataset))
          sampler=SubsetRandomSampler(t_indices)
      
      t_loader = DataLoader(dataset=t_dataset, batch_size=configs['BATCH_SIZE'],
                          sampler = sampler, num_workers=NUM_WORKERS, pin_memory=True, drop_last = True)
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
                try:
                    with open('./testsubjectids.pkl') as f:  # Python 3: open(..., 'rb')
                        testids = pickle.load(f)
                except TypeError:
                    with open('./testsubjectids.pkl', 'rb') as f:
                        testids = pickle.load(f, encoding='latin1')
            else:
                prng = np.random.RandomState()
                queue = prng.permutation(subjectids.shape[0])
                testids = set(subjectids[queue[:int(0.2*subjectids.shape[0])]])
            test_images = cases_to_use.loc[cases_to_use['subjectid'].isin(testids)]
            train_images = cases_to_use.loc[~cases_to_use['subjectid'].isin(testids)]  
            assert(len(pd.merge(test_images, train_images, on=['subjectid'])) == 0)

        logging.info('total images training: '+str(np.array(train_images).shape[0]))
        logging.info('total images test: '+str(np.array(test_images).shape[0]))
        
        train_loader=get_loader(all_images, train_images, all_labels, trainTransformSequence, split = 'train')
        if i == 0:
            eval_train_loader=get_loader(all_images, train_images, all_labels, testTransformSequence, split = 'eval_train', verbose = False)
        
        test_labels = all_labels
        
        if configs['use_only_2017_for_test']:
            test_labels = test_labels[(test_labels['dataset'] =='2017')]
            
        test_loader=get_loader(all_images, test_images, test_labels, testTransformSequence, 'validation')

        logging.info('finished loaders and generators. starting models')
        
        model = model_loading.get_model(num_ftrs)
        
        logging.info('finished models. starting criterion')
        
        def relative_error_mse_loss(input, target):
            return torch.sum(torch.mean((torch.max(input,target)/torch.min(input,target)-1) ** (configs['exponent_relative_error_mse_loss']), dim = 0))
        
        #defining what loss function should be used
        losses_dict = {'l1':nn.L1Loss(size_average = True).cuda(), 
          'l2':nn.MSELoss(size_average = True).cuda(), 
          'smoothl1':nn.SmoothL1Loss(size_average = True).cuda(), 
          'bce':nn.BCELoss(size_average = True).cuda(),
          'relative_mse':relative_error_mse_loss}
        criterion = [{'loss':losses_dict[configs['get_individual_kind_of_loss'][k]], 'weight':configs['get_individual_loss_weights'][k]} for k in configs['get_labels_columns']]
        if configs['optimizer']=='adam':
            optimizer = optim.Adam( [
              {'params':model.module.stn.parameters(), 'lr':configs['initial_lr_location'], 'weight_decay':configs['l2_reg_location']},
              {'params':model.module.final_layers.parameters(), 'lr':configs['initial_lr_fc'], 'weight_decay':configs['l2_reg_fc']}, 
              {'params':model.module.cnn.parameters(), 'lr':configs['initial_lr_cnn'], 'weight_decay':configs['l2_reg_cnn']}
              ], lr=configs['initial_lr_fc'] , weight_decay=configs['l2_reg_fc'])
        elif configs['optimizer']=='nesterov':
            optimizer = optim.SGD( [
              {'params':model.module.stn.parameters(), 'lr':configs['initial_lr_location'], 'weight_decay':configs['l2_reg_location']},
              {'params':model.module.final_layers.parameters(), 'lr':configs['initial_lr_fc'], 'weight_decay':configs['l2_reg_fc']}, 
              {'params':model.module.cnn.parameters(), 'lr':configs['initial_lr_cnn'], 'weight_decay':configs['l2_reg_cnn']}
              ], lr=configs['initial_lr_fc'] , momentum = 0.9, nesterov = True, weight_decay=configs['l2_reg_fc'])
        '''
        if utils.compare_versions(torch.__version__, '0.4.0')>=0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min', verbose = True)
        else:
            scheduler = utils.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min', verbose = True)
        '''
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
    
    if configs['fully_connected_kind']=='dsnm':
        for i in training_range:
                
            model = models[i]
            criterion = criterions[i]
            optimizer = optimizers[i]
            train_loader = train_loaders[i]
            test_loader = test_loaders[i]
            _, _, _, _, _, _ = run_epoch(
                              loaders=eval_train_loader,
                              model=model,
                              criterion=criterion,
                              optimizer=None,
                              epoch=-1,
                              n_epochs=n_epochs,
                              train=False
                            )
            model.module.final_layers.fc_part.initialize()
    
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
            
            if epoch==n_epochs or True:
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
                if epoch >=configs['first_epoch_scheduler_step']:
                    schedulers[i].step(loss)
            all_val_losses = np.concatenate((all_val_losses, loss), axis=0)
            
        logging.info('Train loss ' + str(epoch) + '/' + str(n_epochs) + ': ' +str(np.average(all_train_losses)))
        logging.info('Val loss ' + str(epoch) + '/' + str(n_epochs) + ': '+str(np.average(all_val_losses)))
        
        if epoch==n_epochs or True:
            if configs['save_model'] and epoch==n_epochs:
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
    joined_tables = pd.merge(all_images_merged, all_labels, on=['subjectid', 'crstudy'])
    
    if configs['remove_lung_transplants']:
        joined_tables = joined_tables[(joined_tables['lung_transplant'] ==0)]
        
    joined_tables = joined_tables[(joined_tables['Date_Diff'] < configs['maximum_date_diff'])]
    
    #TODO: severity test
    #joined_tables = joined_tables[(joined_tables['fev1fvc_predrug'] > 0.7) | ((joined_tables['fev1_ratio'] < 0.5) & (joined_tables['fev1fvc_predrug'] < 0.7))]
    #TODO: lung transplant test
    #joined_tables = joined_tables[(joined_tables['lung_transplant'] ==0)]
    cases_to_use = joined_tables[['subjectid', 'crstudy']].groupby(['subjectid', 'crstudy']).count().reset_index()
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
    
    visualize = False
    if visualize:
        load_model(models, 'model20180606-113606-4579')
        for i, model in enumerate(models):
            visualization.GradCAM(model).execute(test_loaders[i])

    train(models, criterions, optimizers, schedulers, train_loaders, test_loaders, eval_train_loader)
    
if __name__ == '__main__':
    main()