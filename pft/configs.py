from future.utils import raise_with_traceback
import logging
import time
import socket 
from future.utils import iteritems
import torch.nn as nn
from random import randint
from functools import partial
from label_preprocessing import PreTransformLabels
class ConfigsClass(object):
    def __init__(self):
        self.configs = {}
        self.predefined = {}
        self.configs_set_values_frozen = False
        self.configs_get_values_frozen = True
        self.configs_add_variables_frozen = False
        self.has_args = {}
        
    def __getitem__(self, name):
        if self.configs_get_values_frozen:
            raise_with_traceback(ValueError('Variables cannot be used yet because they were not freed'))
        return self.get_variable(name)
    
    def __setitem__(self, key, item):
        if self.configs_set_values_frozen:
            raise_with_traceback(ValueError('Variables cannot be changed anymore because they were frozen'))
        if key not in list(self.configs.keys()):
            raise_with_traceback(ValueError('Variable ' + name + 'was not added to configs'))
        self.set_variable(key, item)
        
    def freeze_configs_keys(self):
        self.configs_add_variables_frozen = True
        
    def add_variable(self, name, default, has_args = False):
        if self.configs_add_variables_frozen:
            raise_with_traceback(ValueError('Variables cannot be added anymore because they were frozen'))
        if name in list(self.configs.keys()):
            raise_with_traceback(ValueError('Variable ' + name + 'was already added before'))
        self.has_args[name] = has_args
        self.set_variable(name, default)
    
    def set_variable(self, name, value):
        if callable(value):
            var_as_func = value
        else:
            var_as_func = lambda self: value
        self.configs[name] = partial(var_as_func, self=self)
        
    def get_variable(self, name):
        return (self.configs[name]) if self.has_args[name] else (self.configs[name])()

    def log_configs(self):
        logging.info('-------------------------------used configs-------------------------------')
        for key, value in iteritems(self.configs):
            logging.info(key + ': ' + str(self.get_variable(key)).replace('\n', ' ').replace('\r', ''))
        logging.info('-----------------------------end used configs-----------------------------')
        
    def add_predefined_set_of_configs(self,name, dict_configs):
        if not set(dict_configs.keys()).issubset(self.configs.keys()): 
            raise_with_traceback(ValueError('At least one variable in given dict is not present in the configs'))
        self.predefined[name] = dict_configs
        
    def load_predefined_set_of_configs(self, name):
        for key, value in iteritems(self.predefined[name]):
            self.set_variable(key, value)
    
    def add_self_referenced_variable_from_dict(self,new_variable_name, referenced_variable_name, dict_returns):
        def a(self):
            return dict_returns[self[referenced_variable_name]]
        self.add_variable(new_variable_name,a)
    
    def open_get_block_set(self):
        self.configs_set_values_frozen = True
        self.configs_get_values_frozen = False
    
timestamp = time.strftime("%Y%m%d-%H%M%S")+ '-' + str(randint(1000, 9999))

configs = ConfigsClass()

#configs from this block should probably be the same regardless of model and machine
configs.add_variable('training_pipeline', 'simple') #one_vs_all, ensemble, simple
configs.add_variable('use_set_29', False)
configs.add_variable('pre_transformation', 'none') # 'boxcox', 'log', 'none'
configs.add_variable('individual_pre_transformation', {'copd':'none'})
configs.add_variable('CKPT_PATH', 'model.pth.tar')
configs.add_variable('timestamp', timestamp)
configs.add_variable('output_image_name', 'results' + timestamp +'.png')
#configs.add_variable('output_model_name', 'model' + timestamp + '.pth')
#configs.add_variable('save_model', True)
configs.add_variable('N_EPOCHS', 50)
configs.add_variable('use_lr_scheduler', True)
configs.add_variable('kind_of_loss', 'l2') #'l1' or 'l2', 'smoothl1', or 'bce'
configs.add_variable('individual_kind_of_loss', {'copd':'bce'})
configs.add_variable('positions_to_use', ['PA']) # set of 'PA', 'AP', 'LAT' in a list 
configs.add_variable('initial_lr_fc', 0.00001)
configs.add_variable('initial_lr_cnn', 0.00001)
configs.add_variable('load_image_features_from_file', True)
configs.add_variable('use_fixed_test_set', True)
configs.add_variable('weight_initialization', 'original') # 'xavier', 'original'
configs.add_variable('bias_initialization', 'original') #'constant', 'original'
configs.add_variable('total_ensemble_models', 5) # only used if configs['training_pipeline']=='ensemble'
configs.add_variable('output_copd', False)
configs.add_variable('percentage_labels', ['fev1fvc_pred','fev1fvc_predrug','fvc_ratio','fev1_ratio','fev1fvc_ratio'])
configs.add_variable('use_true_predicted', False)

#These are the main configs to change from default
configs.add_variable('trainable_densenet', False)
configs.add_variable('use_conv11', False)
configs.add_variable('labels_to_use', 'only_absolute') # 'two_ratios', 'three_absolute', 'all_nine' or 'only_absolute'  or 'none', 'fev1fvc_predrug_absolute'
configs.add_variable('use_lateral', False)
configs.add_variable('tie_cnns_same_weights', False)

# configuration of architecture of end of model
configs.add_variable('n_hidden_layers', 2)
configs.add_variable('channels_hidden_layers', 2048) # only used if configs['n_hidden_layers']>0
configs.add_variable('use_dropout_hidden_layers', 0.0) # 0 turn off dropout; 0.25 gives seems to give about the same results as l = 0.05 # only used if configs['n_hidden_layers']>0
configs.add_variable('use_batchnormalization_hidden_layers', False)
configs.add_variable('conv11_channels', 128) # only used if configs['use_conv11']
configs.add_variable('network_output_kind', 'linear') #'linear', 'softplus' , 'sigmoid'
configs.add_variable('individual_output_kind',{'copd':'sigmoid'})

# these configs are modified depending on model and machine
configs.add_variable('machine_to_use', 'dgx' if socket.gethostname() == 'rigveda' else 'titan' if socket.gethostname() =='linux-55p6' or socket.gethostname() == 'titan' else 'other')
configs.add_variable('remove_pre_avg_pool', True)
configs.add_variable('l2_reg_fc', 0.05)
configs.add_variable('l2_reg_cnn', 0.0)

configs.add_variable('sigmoid_safety_constant', 
                     {'fvc_pred':[0.5,1.15], 
                   'fev1_pred':[0.45,1.15], 
                   'fev1fvc_pred':[0.96,1.02], 
                   'fev1_predrug':[0.75,1.15], 
                   'fev1fvc_predrug':[0.75,1.02],
                   'fev1_ratio':[0.65, 1.25],
                   'fvc_predrug':[0.45, 1.25],
                   'fvc_ratio':[0.5, 1.2],
                   'fev1fvc_ratio':[0.75, 1.15],
                   'copd':[1.0,1.0]}
                    )
                     
configs.add_variable('columns_translations',                      
                     {"Subject_Global_ID": "subjectid", 
                        "CRStudy_Local_ID": "crstudy",
                        "PFTExam_Local_ID": "pftid",
                        'Predicted FVC':'fvc_pred',
                        'Predicted FEV1':'fev1_pred',
                        'Predicted FEV1/FVC':'fev1fvc_pred',
                        'Pre-Drug FVC':'fvc_predrug',
                        'Pre-Drug FEV1':'fev1_predrug',
                        'Pre-Drug FEV1/FVC':'fev1fvc_predrug',
                        'Pre-%Pred FVC':'fvc_ratio',
                        'Pre-%Pred FEV1':'fev1_ratio',
                        'Pre-%Pred FEV1/FVC':'fev1fvc_ratio',
                        'TOBACCO_PAK_PER_DY':'packs_per_day',
                        'TOBACCO_USED_YEARS':'years_of_tobacco',
                        'COPD':'copd'})
                     
configs.add_variable('all_input_columns',lambda self: list(self['columns_translations'].values()))


configs.add_predefined_set_of_configs('densenet', { 'trainable_densenet':True, 
                                           'remove_pre_avg_pool':False,
                                             'use_dropout_hidden_layers':0.25})

configs.add_predefined_set_of_configs('p1', { 'use_batchnormalization_hidden_layers': True,
                                             'output_copd': True,
                                             'weight_initialization': 'xavier',
                                             'bias_initialization': 'constant',
                                             'channels_hidden_layers': 1024,
                                             'initial_lr_fc': 0.001,
                                             'use_lateral': True,
                                             'individual_kind_of_loss': {},
                                             'individual_output_kind':{}
                                             })

configs.add_predefined_set_of_configs('p12', { 'use_batchnormalization_hidden_layers': True,
                                             'output_copd': True,
                                             'weight_initialization': 'xavier',
                                             'bias_initialization': 'constant',
                                             'channels_hidden_layers': 1024,
                                             'initial_lr_fc': 0.001,
                                             'use_lateral': True,
                                             'individual_kind_of_loss':{}
                                             })

configs.add_predefined_set_of_configs('p13', { 'use_batchnormalization_hidden_layers': True,
                                             'output_copd': True,
                                             'weight_initialization': 'xavier',
                                             'bias_initialization': 'constant',
                                             'channels_hidden_layers': 512,
                                             'initial_lr_fc': 0.001,
                                             'use_lateral': True
                                             })


configs.add_predefined_set_of_configs('p2', { 'use_batchnormalization_hidden_layers': True,
                                             'output_copd': True,
                                             'weight_initialization': 'xavier',
                                             'bias_initialization': 'constant',
                                             'channels_hidden_layers': 128,
                                             'initial_lr_fc': 0.0001,
                                             'use_lateral': True,
                                             'use_dropout_hidden_layers':0.25,
                                             'l2_reg_fc': 0.0005,
                                             'labels_to_use': 'two_ratios',
                                             'trainable_densenet':True,
                                             'use_conv11': True,
                                             'remove_pre_avg_pool':False,
                                             'n_hidden_layers': 1
                                             })

configs.add_self_referenced_variable_from_dict('get_available_memory', 'machine_to_use',
                                      {'dgx': 15600-550-10, 
                                       'titan':11700-550-10, 
                                       'other':9000-2*550-600}) 
def get_batch_size(self):
    if self['trainable_densenet']:
        return int(self['get_available_memory']/140./(2. if self['use_lateral'] else 1))
    else:
        return 128

def get_individual_characteristic(individual_string, general_string, self):
    #return [(self[individual_string][self['get_labels_columns'][k]] if (self['get_labels_columns'][k] in list(self[individual_string].keys())) else self[general_string]) for k in range(len(self['get_labels_columns']))]
    return {self['all_input_columns'][k]:(self[individual_string][self['all_input_columns'][k]] if (self['all_input_columns'][k] in list(self[individual_string].keys())) else self[general_string]) for k in range(len(self['all_input_columns']))}

def get_individual_kind_of_loss(self):
    return get_individual_characteristic('individual_kind_of_loss', 'kind_of_loss', self)

def get_individual_output_kind(self):
    return get_individual_characteristic('individual_output_kind', 'network_output_kind', self)

def get_individual_pre_transformation(self):
    return get_individual_characteristic('individual_pre_transformation', 'pre_transformation', self)

configs.add_variable('BATCH_SIZE',lambda self: get_batch_size(self))

configs.add_variable('get_individual_kind_of_loss',lambda self: get_individual_kind_of_loss(self))

configs.add_variable('get_individual_output_kind',lambda self: get_individual_output_kind(self))

configs.add_variable('get_individual_pre_transformation',lambda self: get_individual_pre_transformation(self))

configs.add_predefined_set_of_configs('frozen_densenet', {})

configs.add_predefined_set_of_configs('copd_only', {'kind_of_loss':'bce', 
                                                          'labels_to_use':None,
                                                          'network_output_kind':'sigmoid',
                                                          'output_copd':True})

# defining all variables that the network should output
configs.add_self_referenced_variable_from_dict('get_labels_columns_pft', 'labels_to_use',
                                      {'two_ratios': ['fev1fvc_predrug','fev1_ratio'], 
                                       'three_absolute':['fev1_predrug','fvc_predrug', 'fev1_pred'], 
                                       'all_nine':['fvc_pred','fev1_pred','fev1fvc_pred','fvc_predrug','fev1_predrug','fev1fvc_predrug','fvc_ratio','fev1_ratio','fev1fvc_ratio'],
                                       'only_absolute':['fev1_predrug','fvc_predrug', 'fev1_pred', 'fvc_pred'], 
                                       'fev1fvc_predrug_absolute':['fev1_predrug','fvc_predrug'], 
                                       'none':[]}) 
                                      
configs.add_self_referenced_variable_from_dict('get_labels_columns_copd', 'output_copd',
                                      {True: ['copd'], False: []}) 

configs.add_variable('get_labels_columns',lambda self: self['get_labels_columns_pft'] + self['get_labels_columns_copd'])

configs.add_self_referenced_variable_from_dict('pft_plot_columns', 'labels_to_use',
                                      {'two_ratios': [['fev1fvc_predrug'],['fev1_ratio']], 
                                       'three_absolute':[['fev1_predrug','fvc_predrug', 'fev1_pred'],['fev1fvc_predrug'],['fev1_ratio']], 
                                       'all_nine':[['fvc_pred','fev1_pred','fev1fvc_pred','fvc_predrug','fev1_predrug','fev1fvc_predrug','fvc_ratio','fev1_ratio','fev1fvc_ratio']],
                                       'only_absolute':[['fev1_predrug','fvc_predrug', 'fev1_pred', 'fvc_pred'], ['fev1fvc_predrug'],['fev1_ratio']], 
                                       'fev1fvc_predrug_absolute':[['fev1_predrug','fvc_predrug'], ['fev1fvc_predrug']], 
                                       'none':[]})
                                      
configs.add_variable('pre_transform_labels', PreTransformLabels(configs))