from future.utils import raise_with_traceback
import logging
import time
import socket 
from future.utils import iteritems
import torch.nn as nn

class ConfigsClass(object):
    def __init__(self):
        self.configs = {}
        self.functions = {}
        self.predefined = {}
        self.configs_values_frozen = False
        self.configs_keys_frozen = False
        self.variable_function = {}
        
    def __getitem__(self, name):
        return self.configs[name]
    
    def __setitem__(self, key, item):
        if self.configs_values_frozen:
            raise_with_traceback(ValueError('Variables cannot be changed anymore because they were frozen'))
        if key not in list(self.configs.keys()):
            raise_with_traceback(ValueError('Variable ' + name + 'was not added to configs'))
        self.configs[key] = item
        
    def freeze_configs_keys(self):
        self.configs_keys_frozen = True
        
    def add_variable(self, name, default):
        if self.configs_keys_frozen:
            raise_with_traceback(ValueError('Variables cannot be added anymore because they were frozen'))
        if name in list(self.configs.keys()):
            raise_with_traceback(ValueError('Variable ' + name + 'was already added before'))
        self.configs[name] = default
    
    def set_variable(self, name, value):
        self.configs[name] = value
        
    def add_enum_return(self, variable_name, variable_value,function_name, return_function):
        if variable_name not in list(self.configs.keys()):
            raise_with_traceback(ValueError('Variable ' + variable_name + ' was not added to configs'))
        if function_name not in list(self.functions.keys()):
            self.variable_function[function_name] = variable_name
            self.functions[function_name] = {}
        elif not self.variable_function[function_name] == variable_name:
            raise_with_traceback(ValueError('Function ' + function_name + ' was already set for variable ' + self.variable_function[function_name] + ' so it cannot be set for variable ' + variable_name))
        self.functions[function_name][variable_value] = return_function
            
    def get_enum_return(self,function_name):
        a = self.functions[function_name]
        try:
            return a[self.configs[self.variable_function[function_name]]]()
        except KeyError:
            raise_with_traceback(ValueError('Function ' + function_name + ' has no function defined for variable ' + self.variable_function[function_name] + ' with value ' + self.configs[self.variable_function[function_name]]))
        
    def log_configs(self):
        logging.info('-------------------------------used configs-------------------------------')
        for key, value in iteritems(self.configs):
            logging.info(key + ': ' + str(value))
        logging.info('-----------------------------end used configs-----------------------------')
        
    def add_predefined_set_of_configs(self,name, dict_configs):
        if not set(dict_configs.keys()).issubset(self.configs.keys()): 
            raise_with_traceback(ValueError('At least one variable in given dict is not present in the configs'))
        self.predefined[name] = dict_configs
        
    def load_predefined_set_of_configs(self, name):
        for key, value in iteritems(self.predefined[name]):
            self.configs[key] = value
        self.configs_values_frozen = True

timestamp = time.strftime("%Y%m%d-%H%M%S")

configs = ConfigsClass()

#configs from this block should probably be the same regardless of model and machine
configs.add_variable('training_pipeline', 'simple') #one_vs_all, ensemble, simple
configs.add_variable('use_set_29', False)
configs.add_variable('use_log_transformation', False)
configs.add_variable('CKPT_PATH', 'model.pth.tar')
configs.add_variable('timestamp', timestamp)
configs.add_variable('output_image_name', 'results' + timestamp + '.png')
configs.add_variable('N_EPOCHS', 50)
configs.add_variable('use_lr_scheduler', True)
configs.add_variable('kind_of_loss', 'l2') #'l1' or 'l2', 'smoothl1', or 'bce'
configs.add_variable('positions_to_use', ['PA']) # set of 'PA', 'AP', 'LAT' in a list 
configs.add_variable('initial_lr', 0.00001)
configs.add_variable('load_image_features_from_file', True)
configs.add_variable('use_fixed_test_set', True)
configs.add_variable('weight_initialization', 'original') # 'xavier', 'original'
configs.add_variable('bias_initialization', 'original') #'constant', 'original'
configs.add_variable('total_ensemble_models', 5) # only used if configs['training_pipeline']=='ensemble'

#These are the main configs to change from default
configs.add_variable('trainable_densenet', False)
configs.add_variable('use_conv11', False)
configs.add_variable('labels_to_use', 'only_absolute') # 'two_ratios', 'three_absolute', 'all_nine' or 'only_absolute'

# configuration of architecture of end of model
configs.add_variable('n_hidden_layers', 2)
configs.add_variable('channels_hidden_layers', 2048) # only used if configs['n_hidden_layers']>0
configs.add_variable('use_dropout_hidden_layers', 0.0) # 0 turn off dropout; 0.25 gives seems to give about the same results as l = 0.05 # only used if configs['n_hidden_layers']>0
configs.add_variable('use_batchnormalization_hidden_layers', False)
configs.add_variable('conv11_channels', 128) # only used if configs['use_conv11']
configs.add_variable('network_output_kind', 'linear') #'linear', 'softplus' , 'sigmoid'

# these configs are modified depending on model and machine
configs.add_variable('machine_to_use', 'dgx' if socket.gethostname() == 'rigveda' else 'other')
configs.add_variable('remove_pre_avg_pool', True)
configs.add_variable('BATCH_SIZE', 128)
configs.add_variable('l2_reg', 0.05)

configs.add_variable('sigmoid_safety_constant', 
                     {'fvc_pred':[0.5,1.15], 
                   'fev1_pred':[0.45,1.15], 
                   'fev1fvc_pred':[0.96,1.02], 
                   'fev1_predrug':[0.75,1.15], 
                   'fev1fvc_predrug':[0.75,1.02],
                   'fev1_ratio':[0.65, 1.25],
                   'fvc_predrug':[0.45, 1.25],
                   'fvc_ratio':[0.5, 1.2],
                   'fev1fvc_ratio':[0.75, 1.15]}
                    )

configs.freeze_configs_keys()

configs.add_predefined_set_of_configs('densenet', { 'trainable_densenet':True, 
                                           'remove_pre_avg_pool':False,
                                             'BATCH_SIZE': 64 if configs['machine_to_use']=='dgx' else 16,    
                                             'l2_reg':0,
                                             'use_dropout_hidden_layers':0.25})

configs.add_predefined_set_of_configs('frozen_densenet', {})

#defining what loss function should be used
configs.add_enum_return('kind_of_loss', 'l1','get_loss', lambda: nn.L1Loss(size_average = True).cuda())
configs.add_enum_return('kind_of_loss', 'l2','get_loss', lambda: nn.MSELoss(size_average = True).cuda())
configs.add_enum_return('kind_of_loss', 'smoothl1','get_loss', lambda: nn.SmoothL1Loss(size_average = True).cuda())
configs.add_enum_return('kind_of_loss', 'bce','get_loss', lambda: nn.BCELoss(size_average = True).cuda())

# defining all variables that the network should output
configs.add_enum_return('labels_to_use', 'two_ratios','get_labels_columns', 
                lambda: ['fev1fvc_predrug','fev1_ratio'])

configs.add_enum_return('labels_to_use', 'three_absolute','get_labels_columns', 
                lambda: ['fev1_predrug','fvc_predrug', 'fev1_pred'])
                
configs.add_enum_return('labels_to_use', 'all_nine','get_labels_columns', 
                lambda: ['fvc_pred','fev1_pred','fev1fvc_pred','fvc_predrug','fev1_predrug','fev1fvc_predrug','fvc_ratio','fev1_ratio','fev1fvc_ratio'])

configs.add_enum_return('labels_to_use', 'only_absolute','get_labels_columns', 
                lambda: ['fev1_predrug','fvc_predrug', 'fev1_pred', 'fvc_pred'])


# defining what variables are going to be ploted (plot_columns) and how they are calculated
# using the variables present in labels_to_use
def important_ratios_plot_calc(values, k):
    return values[:,0]/values[:,k+1] 

def absolutes_and_important_ratios_plot_calc(values, k):
    if k<4:
        return common_plot_calc(values, k)
    else:
        return important_ratios_plot_calc(values, k-4)
  
def common_plot_calc(values, k):
    return values[:,k]

configs.add_enum_return('labels_to_use', 'two_ratios','get_plot_configs', 
                lambda: {'plot_columns':['fev1fvc_predrug','fev1_ratio'],
                         'plot_function':common_plot_calc})

configs.add_enum_return('labels_to_use', 'three_absolute','get_plot_configs', 
                lambda: {'plot_columns':['fev1fvc_predrug','fev1_ratio'],
                         'plot_function':important_ratios_plot_calc})
                
configs.add_enum_return('labels_to_use', 'all_nine','get_plot_configs', 
                lambda: {'plot_columns':['fvc_pred','fev1_pred','fev1fvc_pred','fvc_predrug','fev1_predrug','fev1fvc_predrug','fvc_ratio','fev1_ratio','fev1fvc_ratio'],
                         'plot_function':common_plot_calc})

configs.add_enum_return('labels_to_use', 'only_absolute','get_plot_configs', 
                lambda: {'plot_columns':['fev1_predrug','fvc_predrug', 'fev1_pred', 'fvc_pred', 'fev1fvc_predrug','fev1_ratio'],
                         'plot_function':absolutes_and_important_ratios_plot_calc})