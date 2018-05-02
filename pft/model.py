#!/home/sci/ricbl/Documents/virtualenvs/dgx_python2_pytorch0.1/bin/python
#SBATCH --time=0-30:00:00 # walltime, abbreviated by -t
#SBATCH --nodes=1 # number of cluster nodes, abbreviated by -N
#SBATCH --mincpus=8
#SBATCH --gres=gpu:1
#SBATCH -o dgx_log/slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e dgx_log/slurm-%j.err-%N # name of the stderr, using the job number (%j) and the first node (%N)
#SBATCH --mem=70G

"""
The main CheXNet model implementation.
"""
import torch
#import torch.multiprocessing as mp
#mp.set_start_method('spawn')
import sys
import os
import time
import re
sys.path.append(os.getcwd())
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import logging
timestamp = time.strftime("%Y%m%d-%H%M%S")
logging.basicConfig( filename = 'log/log'+timestamp+'.txt', level=logging.INFO)
print('log'+timestamp+'.txt')
import numpy as np
np.set_printoptions(threshold=np.inf)
from torch.nn.modules import Module
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torch.utils.model_zoo as model_zoo
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from random import *
import pandas as pd
from PIL import Image
from PIL import ImageMath
from future.utils import raise_with_traceback
from future.utils import iteritems
from h5df import Store
import socket 

import h5py

try:
    import cPickle as pickle
except:
    import _pickle as pickle 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from functools import partial

logging.info('Using PyTorch ' +str(torch.__version__))

percentage_labels = ['fev1fvc_pred','fev1fvc_predrug','fvc_ratio','fev1_ratio','fev1fvc_ratio']

configs = {}

NUM_WORKERS = 1

#configs from this block should probably be the same regardless of model and machine
configs['one_vs_all'] = False
configs['use_set_29'] = False
configs['use_log_transformation'] = False
configs['CKPT_PATH'] = 'model.pth.tar'
configs['output_image_name'] = 'resultsmse' + timestamp + '.png'
configs['N_EPOCHS'] = 50
configs['use_lr_scheduler'] = True
configs['kind_of_loss'] = 'l2' #'l1' or 'l2' or 'smoothl1'
configs['positions_to_use'] = ['PA']
configs['initial_lr'] = 0.00001
configs['load_image_features_from_file'] = True

#These are the main configs to change from default
configs['trainable_densenet'] = True
configs['use_conv11'] = False
configs['labels_to_use'] = 'only_absolute' # 'two_ratios', 'three_absolute', 'all_nine' or 'only_absolute'

# configuration of architecture of end of model
configs['n_hidden_layers'] = 2
configs['channels_hidden_layers'] = 2048 # only used if configs['n_hidden_layers']>0
configs['use_dropout_hidden_layers'] = 0.25#0  # 0 turn off dropout # only used if configs['n_hidden_layers']>0
configs['conv11_channels'] = 128 # only used if configs['use_conv11']

# these configs are modified depending on model and machine
configs['machine_to_use'] = 'dgx' if socket.gethostname() == 'rigveda' else 'other'
configs['remove_pre_avg_pool'] = False if configs['trainable_densenet'] else True
configs['BATCH_SIZE'] = 64 if (configs['machine_to_use']=='dgx' and configs['trainable_densenet']) else (16 if (configs['trainable_densenet']) else (128))
configs['l2_reg'] = 0#.05 # 0 if configs['trainable_densenet'] else 0.05 #0.0005

def important_ratios_plot_calc(values, k):
    return values[:,0]/values[:,k+1] 

def absolutes_and_important_ratios_plot_calc(values, k):
    if k<4:
        return common_plot_calc(values, k)
    else:
        return important_ratios_plot_calc(values, k-4)
  
def common_plot_calc(values, k):
    return values[:,k]

if configs['labels_to_use']=='two_ratios':
    labels_columns = ['fev1fvc_predrug','fev1_ratio']
    plot_columns = labels_columns
    plot_function = common_plot_calc
elif configs['labels_to_use']=='three_absolute':
    labels_columns = ['fev1_predrug','fvc_predrug', 'fev1_pred']
    plot_columns = ['fev1fvc_predrug','fev1_ratio']
    plot_function = important_ratios_plot_calc
elif configs['labels_to_use']=='all_nine':
    labels_columns = ['fvc_pred','fev1_pred','fev1fvc_pred','fvc_predrug','fev1_predrug','fev1fvc_predrug','fvc_ratio','fev1_ratio','fev1fvc_ratio']
    plot_columns = labels_columns
    plot_function = common_plot_calc
elif configs['labels_to_use']=='only_absolute':
    labels_columns = ['fev1_predrug','fvc_predrug', 'fev1_pred', 'fvc_pred']
    plot_columns = ['fev1_predrug','fvc_predrug', 'fev1_pred', 'fvc_pred', 'fev1fvc_predrug','fev1_ratio']
    plot_function = absolutes_and_important_ratios_plot_calc
else:
    raise_with_traceback(ValueError('configs["labels_to_use"] was set to an invalid value: ' + configs['labels_to_use']))

def get_name_pickle_file():
    if configs['trainable_densenet']:
        size_all_images = '224_224'
    elif configs['remove_pre_avg_pool']:
        size_all_images = '7_7'
    else:
        size_all_images = '1_1'
    return 'all_images_' +  size_all_images + '_prot2.pkl'

class DatasetGenerator(Dataset):
    def __init__ (self, pathDatasetFile, transform = None):
        super(DatasetGenerator, self).__init__()
        self.listImage = pathDatasetFile
        
        if configs['trainable_densenet'] and configs['load_image_features_from_file']:
            self.file = h5py.File('./all_images_224_224_prot2.2.h5', 'r')
            
            #much slower
            #store = Store('all_images_224_224_prot2.2.h5df', mode="r")
            #self.df1 = store["/frames/1"]
        else: 
            self.n_images = len(self.listImage)
        self.n_images = len(self.listImage)
        self.transform = transform
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        if (configs['trainable_densenet'] and (not configs['load_image_features_from_file'])):
            imagePath = self.listImage['filepath'].iloc[index]
            imageData = Image.open(imagePath)
            
        else:
            if configs['trainable_densenet']:
                old_index = self.listImage['preprocessed'].iloc[index]
                
                imageData = self.file['dataset_1'][old_index,...].astype('float32')
                
                #much slower
                #examples_to_read = ['ind' + str(old_index)]
                #imageData = self.df1.rows(examples_to_read).values.reshape(3,224,224)
            else:
                imageData = self.listImage['preprocessed'].iloc[index][0]
        #a = self.listImage[labels_columns]
        #print(index)
        #print(len(a))
        #print(self.listImage['filepath'].iloc[index])
        #imageData = np.flip(imageData,1)
        #b = np.transpose((imageData), axes = (1,2,0))*[[[0.229, 0.224, 0.225]]]+[[[0.485, 0.456, 0.406]]] 
        #b = b - [[[np.amin(b)]]]
        #print(np.amax(b))
        #print(np.amin(b))
        #plt.imshow(b)
        #plt.savefig('test.png')
        #1/0
        imageLabel= torch.FloatTensor(self.listImage[labels_columns].iloc[index])
        #self.transform = transforms.ToTensor()
        if self.transform is not None: 
            imageData = self.transform(imageData)
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        return self.n_images
    
 #-------------------------------------------------------------------------------- 
 
def write_filenames():
    pathFileTrain = '/home/sci/ricbl/Documents/projects/radiology-project/pft/train2.txt'
    pathPngs = '/usr/sci/projects/DeepLearning/Tolga_Lab/Projects/Project_JoyceSchroeder/data/data_PFT/'
    command = 'find ' + pathPngs + ' -type f -name "*.png" > ' + pathFileTrain
    os.system(command)

def read_filenames(pathFileTrain):
    listImage = []
    fileDescriptor = open(pathFileTrain, "r")
    
    line = True
    while line:             
      line = fileDescriptor.readline()

      #--- if not empty
      if line:
          thisimage = {}
          lineItems = line.split()
          thisimage['filepath'] = lineItems[0]
          splitted_filepath = thisimage['filepath'].replace('\\','/').split("/")
          splitted_ids = splitted_filepath[-1].replace('-','_').replace('.','_').split('_')
          thisimage['subjectid'] = int(splitted_ids[1])
          thisimage['crstudy'] = int(splitted_ids[3])
          thisimage['scanid'] = int(splitted_ids[5])
          
          position = splitted_ids[-2].upper()
          if 'LAT' in position:
              position = 'LAT'
          elif 'PA' in position:
              position = 'PA'
          elif 'AP' in position:
              position = 'AP'
          elif 'LARGE' in position:
              continue
          elif 'SUPINE' in position:
              continue
          elif 'CHEST' in position:
              continue
          elif 'P' == position and  splitted_ids[-3].upper() == 'A':
              position = 'AP'
          elif 'PORTRAIT' in position:
              continue
          else:
              raise_with_traceback(ValueError('Unknown position: '+position + ', for file: ' +  lineItems[0]))
          thisimage['position'] = position
          listImage.append(thisimage)
    fileDescriptor.close()
    return pd.DataFrame(listImage)

class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. Default: 10.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    logging.info('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = (-float('inf'))

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)
        
class Meter():
    """
    A little helper class which keeps track of statistics during an epoch.
    """
    def __init__(self, name, cum=False):
        self.cum = cum
        if type(name) == str:
            name = (name,)
        self.name = name

        self._total = torch.zeros(len(self.name))
        self._last_value = torch.zeros(len(self.name))
        self._count = 0.0


    def update(self, data, n=1):
        self._count = self._count + n
        if isinstance(data, torch.autograd.Variable):
            self._last_value.copy_(data.data)
        elif isinstance(data, torch.Tensor):
            self._last_value.copy_(data)
        else:
            self._last_value.fill_(data)
        self._total.add_(self._last_value)


    def value(self):
        if self.cum:
            return self._total
        else:
            return self._total / self._count


    def __repr__(self):
        return '\t'.join(['%s: %.5f (%.3f)' % (n, lv, v)
            for n, lv, v in zip(self.name, self._last_value, self.value())])
      
def run_epoch(loader, model, criterion, optimizer, epoch=0, n_epochs=0, train=True):
    time_meter = Meter(name='Time', cum=True)
    loss_meter = Meter(name='Loss', cum=False)
    error_meter = Meter(name='Error', cum=False)
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
        #logging.info(input.size())
        #logging.info(input_var.size())
        output_var = model(input_var)
        loss = criterion(output_var, target_var)
        if train:
            train_string = 'train'
        else:
            train_string = 'val'
        if (not train) or True:
            if epoch==n_epochs:
                if configs['use_log_transformation']:
                    output_var_fixed = np.exp(output_var.data.cpu().numpy())
                    target_var_fixed = np.exp(target_var.data.cpu().numpy())
                else:
                    output_var_fixed = output_var.data.cpu().numpy()
                    target_var_fixed = target_var.data.cpu().numpy()
                    
                y_corr = np.concatenate((y_corr, target_var_fixed), axis = 0)
                y_pred = np.concatenate((y_pred, output_var_fixed), axis = 0)
                    
                #logging.info(output_var_fixed)
                #logging.info(target_var_fixed)
                #print(output_var.data.cpu().numpy())
                #print(target_var.data.cpu().numpy())
                #logging.info(output_var_fixed.shape)
                #logging.info(target_var_fixed.shape)
                
                
                
                markers = ['b.','g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'b>', 'g>','r>']
                plot_var_for_legend = []
                
                for k in range(len(plot_columns)):
                    this_plot, = plt.plot(plot_function(target_var_fixed, k),plot_function(output_var_fixed, k), markers[k])
                    plot_var_for_legend.append(this_plot)
                #plt.legend(plot_var_for_legend, labels_columns,
                plt.legend(plot_var_for_legend, plot_columns,
                            scatterpoints=1,
                            loc='best',
                            ncol=2,
                            fontsize=8)
                
                
        # Backward pass
        if train:
            loss.backward()
            optimizer.step()
            optimizer.n_iters = optimizer.n_iters + 1 if hasattr(optimizer, 'n_iters') else 1
        # Accounting
        predictions_var = torch.ge(output_var, 0.5)
        error = 1 - torch.eq(predictions_var, target_var.byte()).float().mean()
        
        
        batch_time = time.time() - end
        end = time.time()

        # Log errors

        time_meter.update(batch_time)
        loss_meter.update(loss)
        if not train:
            pass
        lossMean += len(input)*loss.data.cpu().numpy()
        #lossMean += len(input)*loss.data[0]
        lossValNorm += len(input)
        '''
        error_meter.update(error)
        logging.info('  '.join([
            '%s: (Epoch %d of %d) [%04d/%04d]' % ('Train' if train else 'Eval',
                epoch, n_epochs, i + 1, len(loader)),
            str(time_meter),
            str(loss_meter),
            str(error_meter),
        ]))
        '''
    if epoch==n_epochs:
        r2s = {}
        for k in range(len(plot_columns)):
            r2s[plot_columns[k]] = r2(y_corr = plot_function(y_corr,k), y_pred = plot_function(y_pred,k))
        logging.info('r2: ' + str(r2s))
        
        ax = plt.gca()
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        final_axis = (min(ylim[0],xlim[0]), max(ylim[1],xlim[1]))
        ax.set_ylim(final_axis)
        ax.set_xlim(final_axis)
        plt.plot([final_axis[0], final_axis[1]], [final_axis[0], final_axis[1]], 'k-', lw=1)
        plt.xlabel('Groundtruth', fontsize=10)
        plt.ylabel('Predicted', fontsize=10)
        plt.savefig('plots/' + train_string + configs['output_image_name'])
        plt.clf()
        plt.cla()
        plt.close()
    return time_meter.value(), loss_meter.value(), error_meter.value(), lossMean/lossValNorm

class CropBiggestCenteredInscribedSquare(object):    
    def __init__(self):
        pass

    def __call__(self, tensor):
        longer_side = min(tensor.size)
        horizontal_padding = (longer_side - tensor.size[0]) / 2
        vertical_padding = (longer_side - tensor.size[1]) / 2
        return tensor.crop(
            (
                -horizontal_padding,
                -vertical_padding,
                tensor.size[0] + horizontal_padding,
                tensor.size[1] + vertical_padding
            )
        )

    def __repr__(self):
        return self.__class__.__name__ + '()'
      
class Convert16BitToFloat(object):    
    def __init__(self):
        pass

    def __call__(self, tensor):
        tensor.mode = 'I'
        return ImageMath.eval('im/256', {'im':tensor}).convert('RGB')
        return tensor.point(lambda i:i*(1./256)).convert('RGB')

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomHorizontalFlipNumpy(object):    
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, tensor):
        if random() < self.p:
            #return np.flip(tensor, 2)
            return tensor[:,:,::-1].copy()
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomHorizontalFlipTensor(object):    
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, tensor):
        if random() < self.p:
            idx = [i for i in range(tensor.size(2)-1, -1, -1)]
            
            idx = torch.LongTensor(idx)
            inverted_tensor = tensor.index_select(2, idx)

            return inverted_tensor
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'
      
class ChexnetEncode(object):    
    def __init__(self, model):
        self.model = model

    def __call__(self, tensor):
        input_var = torch.autograd.Variable(tensor.view(1,3,224,224), volatile=True)
        out = self.model(input_var)
        return [np.squeeze(np.transpose(out.data.cpu().numpy(), (0,1,2,3)),axis = 0)]

    def __repr__(self):
        return self.__class__.__name__ + '()'    

def preprocess_image(imagePath, transform):
    imageData = Image.open(imagePath)
    return transform(imageData)
        
def preprocess_images(all_images, transformations):
    a = all_images.apply(lambda row: preprocess_image(row['filepath'], transformations),axis=1)
    all_images['preprocessed'] = a
    return all_images

def sample(iterator, k):
    """
    Samples k elements from an iterable object.

    :param iterator: an object that is iterable
    :param k: the number of items to sample
    """
    # fill the reservoir to start
    result = [next(iterator) for _ in range(k)]

    n = k - 1
    for item in iterator:
        n += 1
        s = randint(0, n)
        if s < k:
            result[s] = item

    return result
  
def log_configs():
    logging.info('-------------------------------used configs-------------------------------')
    for key, value in iteritems(configs):
        logging.info(key + ': ' + str(value))
    logging.info('-----------------------------end used configs-----------------------------')

def cmp(a, b):
    return (a > b) - (a < b)
  
def compare_versions(version1, version2):
    def normalize(v):
        return [int(x) for x in re.sub(r'(\.0+)*$','', v).split(".")]
    return cmp(normalize(version1), normalize(version2))
  
def load_checkpoint(model):
    if os.path.isfile(configs['CKPT_PATH']):
        logging.info("=> loading checkpoint")
        checkpoint = torch.load(configs['CKPT_PATH'])
        state_dict = checkpoint['state_dict']
        
        if compare_versions(torchvision.__version__, '0.2.1')>=0:
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        model.load_state_dict(state_dict)
        logging.info("=> loaded checkpoint")
    else:
        logging.info("=> no checkpoint found")
    return model

def load_pretrained_chexnet():
    chexnetModel = DenseNet121(14).cuda()
    chexnetModel = torch.nn.DataParallel(chexnetModel).cuda()
    chexnetModel = load_checkpoint(chexnetModel)
    return chexnetModel
    
def r2(y_corr, y_pred):
    y_corr_mean = np.mean(y_corr)
    sstot = np.sum(np.square(y_corr-y_corr_mean))
    ssres = np.sum(np.square(y_pred-y_corr))
    return 1-ssres/sstot

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
      
def get_model(num_ftrs):
    # initialize and load the model

    model = torch.nn.Sequential()
    current_n_channels = num_ftrs
    if configs['use_conv11']:
        model.add_module("flatten",nn.Conv2d( in_channels = 1024, out_channels =  configs['conv11_channels'], kernel_size = 1).cuda())
        model.add_module("flatten",nn.ReLU().cuda())
        current_n_channels = configs['conv11_channels']*(49 if configs['remove_pre_avg_pool'] else 1)
    
    model.add_module("flatten",Flatten().cuda())
    if configs['n_hidden_layers'] > 0:
        model.add_module("drop_0",nn.Dropout(p=configs['use_dropout_hidden_layers']).cuda())
        #model.add_module("bn_0",torch.nn.BatchNorm1d(current_n_channels).cuda())
        model.add_module("linear_0",nn.Linear(current_n_channels, configs['channels_hidden_layers'] ).cuda())
        model.add_module("relu_0",nn.ReLU().cuda())
        current_n_channels = configs['channels_hidden_layers']
        for layer_i in range(1, configs['n_hidden_layers']): 
            model.add_module("drop_"+str(layer_i),nn.Dropout(p=configs['use_dropout_hidden_layers']).cuda())
            #model.add_module("bn_"+str(layer_i),torch.nn.BatchNorm1d(current_n_channels).cuda())
            model.add_module("linear_"+str(layer_i),nn.Linear(current_n_channels, configs['channels_hidden_layers'] ).cuda())
            model.add_module("relu_"+str(layer_i),nn.ReLU().cuda())
            current_n_channels = configs['channels_hidden_layers']
          
    model.add_module("linear_out", nn.Linear(current_n_channels , len(labels_columns)).cuda())
    if configs['trainable_densenet']:
        outmodel = load_pretrained_chexnet()
        
        #for param in model.parameters():
        #    param.requires_grad = False         
        
        outmodel.module.densenet121.classifier = model
        
        outmodel = outmodel.cuda()
    else:
        outmodel = model
    return outmodel

def main():
    cudnn.benchmark = False
    log_configs()
    if configs['use_set_29']:
        file_with_image_filenames = 'train.txt'
        file_with_labels = 'labels.csv'
    else:
        file_with_image_filenames = 'train2.txt'
        file_with_labels = 'labels20180420.csv'
    
    logging.info('started feature loading')
    
    num_ftrs = 50176
    if not configs['remove_pre_avg_pool']:
        num_ftrs = int(num_ftrs/7/7)
    
    if (not configs['load_image_features_from_file']) or configs['trainable_densenet']:
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
        list_pretransforms =[     Convert16BitToFloat(),
                  CropBiggestCenteredInscribedSquare(),
                  transforms.Resize(size=(224)), 
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  normalize]
        
        all_images = read_filenames('/home/sci/ricbl/Documents/projects/radiology-project/pft/' + file_with_image_filenames)
       
    if (not configs['trainable_densenet']) and (not configs['load_image_features_from_file']):
        chexnetModel = load_pretrained_chexnet()
        num_ftrs = chexnetModel.module.densenet121.classifier[0].in_features*7*7
        if not configs['remove_pre_avg_pool']:
            num_ftrs = int(num_ftrs/7/7)
        
        chexnetModel.module.densenet121.classifier = nn.Sequential()
        chexnetModel = chexnetModel.cuda()  
        list_pretransforms.append(ChexnetEncode(chexnetModel))
            
        transformSequence = transforms.Compose(list_pretransforms)
        
        all_images = preprocess_images(all_images, transformSequence)
        del transformSequence
        del chexnetModel
        with open(get_name_pickle_file(), 'wb') as f:
            pickle.dump(all_images, f, protocol=2)
    elif (not configs['trainable_densenet']) and (configs['load_image_features_from_file']):
        all_images = pd.read_pickle(get_name_pickle_file())

    logging.info('ended feature loading. started label loading')
    all_labels = pd.read_csv('./' + file_with_labels)
    logging.info('ended label loading')
    
    all_labels[percentage_labels] = all_labels[percentage_labels] /100.
    if configs['use_log_transformation']:
        all_labels[labels_columns] = all_labels[labels_columns].apply(np.log)
    if (configs['trainable_densenet'] and (configs['load_image_features_from_file'])):
        all_images['preprocessed'] = all_images.index
    all_examples = pd.merge(all_images, all_labels, on=['subjectid', 'crstudy'])
    
    pa_images = all_examples[all_examples['position'].isin(configs['positions_to_use'])]
    
    
        
    models = []
    criterions = []
    optimizers = []
    schedulers = []
    train_loaders = []
    test_loaders = []
    
    if configs['one_vs_all']:
        training_range = pa_images['subjectid'].unique()
    else:
        training_range = range(1)
    
    for i in training_range:
        if configs['one_vs_all']:
            train_images = pa_images[pa_images['subjectid']!=i]  
            test_images = pa_images[pa_images['subjectid']==i]
        else:
          
            subjectids = np.array(pa_images['subjectid'].unique())
            queue = np.random.permutation(subjectids.shape[0])
            testids = set(subjectids[queue[:int(0.2*subjectids.shape[0])]])
            test_images = pa_images.loc[pa_images['subjectid'].isin(testids)]
            train_images = pa_images.loc[~pa_images['subjectid'].isin(testids)]  
        
        if configs['trainable_densenet'] and (not configs['load_image_features_from_file']):
            transformSequence = transforms.Compose(list_pretransforms)
        elif configs['trainable_densenet']:
            transformSequence = transforms.Compose([RandomHorizontalFlipNumpy()])
        else: 
            transformSequence = transforms.Compose([RandomHorizontalFlipTensor()])
        logging.info('size training: '+str(np.array(train_images).shape[0]))
        logging.info('size test: '+str(np.array(test_images).shape[0]))
        
        train_dataset = DatasetGenerator(train_images, transformSequence)
        
        train_indices = torch.randperm(len(train_dataset))  
        train_loader = DataLoader(dataset=train_dataset, batch_size=configs['BATCH_SIZE'],
                                sampler=SubsetRandomSampler(train_indices), num_workers=NUM_WORKERS, pin_memory=True, drop_last = True)                          
        
        test_dataset = DatasetGenerator(test_images, transformSequence)
        test_loader = DataLoader(dataset=test_dataset, batch_size=configs['BATCH_SIZE'],
                                shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        logging.info('finished loaders and generators. starting models')
        
        model = get_model(num_ftrs)
        
        logging.info('finished models. starting criterion')
        
        if configs['kind_of_loss'] == 'l1':
            criterion = nn.L1Loss(size_average = True).cuda()
        elif configs['kind_of_loss'] == 'l2':
            criterion = nn.MSELoss(size_average = True).cuda()
        elif configs['kind_of_loss'] == 'smoothl1':
            nn.SmoothL1Loss(size_average = True).cuda()
        else :
            raise_with_traceback(ValueError('configs["kind_of_loss"] was set to an invalid value: ' + str(configs['kind_of_loss'])))
              
        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=configs['initial_lr'] , weight_decay=configs['l2_reg'])
        optimizer = optim.Adam( model.parameters(), lr=configs['initial_lr'] , weight_decay=configs['l2_reg'])
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min', verbose = True)
        models = models + [model]
        criterions = criterions + [criterion]
        optimizers = optimizers + [optimizer]
        schedulers = schedulers + [scheduler]
        train_loaders = train_loaders + [train_loader]
        test_loaders = test_loaders + [test_loader]
        
    # Train model
    
    n_epochs = configs['N_EPOCHS']
    logging.info('starting training')
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
        

            #_set_lr(optimizer, epoch, n_epochs, lr)
            train_results = run_epoch(
                loader=train_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                epoch=epoch,
                n_epochs=n_epochs,
                train=True,
            ) 
            loss = np.atleast_1d( train_results[3])
            all_train_losses = np.concatenate((all_train_losses, loss), axis=0)
            val_results = run_epoch(loader=test_loader,
                model=model,
                criterion=criterion,
                optimizer=None,
                epoch=epoch,
                n_epochs=n_epochs,
                train=False)
            loss = np.atleast_1d( val_results[3])
            if configs['use_lr_scheduler']:
                schedulers[i].step(loss)
            all_val_losses = np.concatenate((all_val_losses, loss), axis=0)
        logging.info('Train loss ' + str(epoch) + '/' + str(n_epochs) + ': ' +str(np.average(all_train_losses)))
        logging.info('Val loss ' + str(epoch) + '/' + str(n_epochs) + ': '+str(np.average(all_val_losses)))

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        
        '''
        self.densenet121 = DenseNetEfficientMulti(growth_rate=32, block_config=(6,12,24,16), compression=0.5,
                 num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=1000, cifar=False)
        
        self.densenet121.load_state_dict(model_zoo.load_url(model_urls['densenet121']))
        '''
        self.densenet121 = torchvision.models.densenet121(pretrained=False)
        
        num_ftrs = self.densenet121.classifier.in_features
        
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        if configs['remove_pre_avg_pool']:
            x = self.densenet121.features(x)
            x = F.relu(x, inplace=True) # should I always have this relu on?
            x = self.densenet121.classifier(x)
        else:
            x = self.densenet121(x)
        return x   

if __name__ == '__main__':
    main()