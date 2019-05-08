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
        for key, value in sorted(iteritems(self.configs)):
            logging.info(key + ': ' + str(self.get_variable(key)).replace('\n', ' ').replace('\r', ''))
        logging.info('-----------------------------end used configs-----------------------------')

    def add_predefined_set_of_configs(self,name, dict_configs):
        if not set(dict_configs.keys()).issubset(self.configs.keys()):
            raise_with_traceback(ValueError('At least one variable in given dict is not present in the configs:' + str(set(dict_configs.keys()).difference(self.configs.keys()))))
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
configs.add_variable('data_to_use', ['2017'])
configs.add_variable('use_images_with_position_LARGE', False)
configs.add_variable('pre_transformation', 'none') # 'boxcox', 'log', 'none', 'residual'
configs.add_variable('individual_pre_transformation', {'copd':'none'})
configs.add_variable('CKPT_PATH', 'densenet121.pth.tar')
configs.add_variable('timestamp', timestamp)
configs.add_variable('output_image_name', 'results' + timestamp +'.png')
configs.add_variable('output_model_name', 'model' + timestamp )
configs.add_variable('save_model', False)
configs.add_variable('N_EPOCHS', 50)
configs.add_variable('use_lr_scheduler', True)
configs.add_variable('kind_of_loss', 'l2') #'l1' or 'l2', 'smoothl1', or 'bce' or 'relative_mse'
configs.add_variable('exponent_relative_error_mse_loss', 1)
configs.add_variable('individual_kind_of_loss', {'copd':'bce'})
configs.add_variable('individual_loss_weights', {'copd':0.33})
configs.add_variable('loss_weight', 1.0)
configs.add_variable('positions_to_use', ['PA']) # set of 'PA', 'AP' in a list
configs.add_variable('initial_lr_fc', 0.00001)
configs.add_variable('initial_lr_cnn', 0.00001)
configs.add_variable('initial_lr_location', 0.0001)
configs.add_variable('first_epoch_scheduler_step', 10)
configs.add_variable('load_image_features_from_file', True)
configs.add_variable('use_fixed_test_set', True)
configs.add_variable('weight_initialization', 'original') # 'xavier', 'original'
configs.add_variable('bias_initialization', 'original') #'constant', 'original'
configs.add_variable('total_ensemble_models', 6) # only used if configs['training_pipeline']=='ensemble'
configs.add_variable('output_copd', False)
configs.add_variable('output_gold', False)
configs.add_variable('percentage_labels', ['fev1fvc_pred','fev1fvc_predrug','fvc_ratio','fev1_ratio','fev1fvc_ratio'])
configs.add_variable('use_true_predicted', False)
configs.add_variable('use_copd_definition_as_label', False)
configs.add_variable('use_random_crops',False)
configs.add_variable('use_extra_inputs',False)
configs.add_variable('dropout_batch_normalization_last_layer',False)
configs.add_variable('layer_to_insert_extra_inputs',lambda self:self['n_hidden_layers']+1)
configs.add_variable('densenet_dropout',0.0) # 0 turn off dropout
configs.add_variable('fc_activation','relu')
configs.add_variable('chexnet_layers',121)
configs.add_variable('chexnet_architecture','densenet')
configs.add_variable('multiplier_constant_meanvar_var_loss',12)#50)#4)
configs.add_variable('multiplier_constant_meanvar_bce_loss',0.01)#)10)#)10/30./100.)
configs.add_variable('multiplier_constant_meanvar_mean_loss',1)
configs.add_variable('use_mean_var_loss',False)
configs.add_variable('meanvarloss_discretization_spacing',0.01)
configs.add_variable('dsnm_n_groups',50)
configs.add_variable('use_pca_dsnm',True)
configs.add_variable('dsnm_pca_size',100)
configs.add_variable('fully_connected_kind','fully_connected') #'fully_connected', 'dsnm', 'softmax_gate'
configs.add_variable('use_more_one_gpu',False)
configs.add_variable('classes_hidden_layers',15)
configs.add_variable('gate_uniformity_loss_multiplier',0.4)
configs.add_variable('mutual_exclusivity_loss_multiplier',0.4)
configs.add_variable('gate_orthogonal_loss_multiplier',1.0)
configs.add_variable('use_spatial_transformer_network',False)
configs.add_variable('use_batchnormalization_location',True)
configs.add_variable('use_dropout_location',0.0)
configs.add_variable('channels_location',32)
configs.add_variable('pretrain_kind', 'chestxray')
configs.add_variable('maximum_date_diff', 180)
configs.add_variable('use_only_2017_for_test', False)
configs.add_variable('remove_lung_transplants', False)
configs.add_variable('balance_dataset_by_fvcfev1_predrug', False)
configs.add_variable('optimizer', 'adam')
configs.add_variable('splits_to_use', 'test_with_val') #'test_with_val', 'test_with_test', 'include_val_in_training' , 'include_test_in_training'
configs.add_variable('use_lung_transplant_in_test', False)
configs.add_variable('max_date_diff_to_use_for_test', 2)
configs.add_variable('remove_repeated_pfts', False)
configs.add_variable('remove_repeated_images', False)
configs.add_variable('create_csv_from_dataset', False)
configs.add_variable('remove_cases_more_one_image_per_position', False)
configs.add_variable('load_model', False)
configs.add_variable('model_to_load', '')
configs.add_variable('prefix_model_to_load', '')
configs.add_variable('skip_train', False)
configs.add_variable('use_sigmoid_channel', False)
configs.add_variable('n_channels_local_convolution', 512)
configs.add_variable('use_local_conv', False)
configs.add_variable('milestones_steps', [35,45])
configs.add_variable('scheduler_to_use', 'plateau')
configs.add_variable('first_parameter_cnn_not_to_freeze', 'conv1')
configs.add_variable('override_max_axis_graph', None)
configs.add_variable('relative_noise_to_add_to_label',None)
configs.add_variable('use_horizontal_flip',False)
configs.add_variable('gamma_range_augmentation',None)
configs.add_variable('degree_range_augmentation',None)
configs.add_variable('scale_range_augmentation',None)
configs.add_variable('histogram_equalization','none') #'none', 'global', 'local'
configs.add_variable('use_delayed_lateral_pooling',False)
configs.add_variable('use_unet_segmentation',False)
configs.add_variable('use_unet_segmentation_for_lateral',False)
configs.add_variable('initial_lr_unet', 0.0)
configs.add_variable('l2_reg_unet', 0.0)
configs.add_variable('unet_multiply_instead_of_channel', False)
configs.add_variable('unet_model_file', 'unet-epoch100-20181012-011958-7421')
configs.add_variable('normalization_segmentation', False)
configs.add_variable('register_with_segmentation', False)
configs.add_variable('calculate_segmentation_features', False)
configs.add_variable('extra_histogram_equalization_for_segmentation', True)
configs.add_variable('vgg_batch_norm', False)
configs.add_variable('squeezenet_version_11', True)
configs.add_variable('magnification_input', 1)
configs.add_variable('use_half_lung', False)
configs.add_variable('segmentation_in_loading', True)
configs.add_variable('create_csv_for_segmentation_features', False)
configs.add_variable('normalize_lateral_and_frontal_with_bn', False)
configs.add_variable('normalize_extra_inputs_and_rest_with_bn', False)
configs.add_variable('extra_fc_layers_for_extra_input', False)
configs.add_variable('extra_fc_layers_for_extra_input_output_size', 256)
configs.add_variable('prevent_train_shuffle', False)
configs.add_variable('use_precalculated_segmentation_features', True)
configs.add_variable('use_log_quotient_output', False)
configs.add_variable('use_binary_classifiers', False)
configs.add_variable('multiplier_constant_binary_classifiers_mean_loss', 1.0)
configs.add_variable('multiplier_constant_binary_classifiers_bce_loss', 1.0)
configs.add_variable('multiplier_constant_binary_classifiers_consistency_loss', 0.0)
configs.add_variable('n_binary_classifiers_when_percentile', 25)
configs.add_variable('binary_classifiers_percentile_spacing', True)
configs.add_variable('binary_classifiers_use_borders', True)
configs.add_variable('binary_classifiers_use_weights', False)
configs.add_variable('binary_classifiers_n_post_layers', 0)
configs.add_variable('binary_classifiers_fit_type', 'sigmoid')
configs.add_variable('plateau_patience', 5)
configs.add_variable('local_directory', '/home/sci/ricbl/Documents/projects/temp_radiology/radiology-project/pft')
configs.add_variable('normalization_mean', [0.485, 0.456, 0.406])
configs.add_variable('normalization_std', [0.229, 0.224, 0.225])
configs.add_variable('post_binary_classifiers_fit_or_linear', 'fit')
configs.add_variable('use_set_spiromics', False)
configs.add_variable('use_transformation_loss', False)
configs.add_variable('transformation_n_groups', 10)
configs.add_variable('transformation_group_size', 10)
configs.add_variable('chestxray14_path', '/usr/sci/projects/DeepLearning/Tolga_Lab/data_public/NIH_ChestX-Ray/images/all_images')
configs.add_variable('transformation_loss_multiplier', 0.1)
configs.add_variable('use_chexpert', False)
configs.add_variable('chestxpert_path', '/usr/sci/projects/DeepLearning/Ricardo_Dataset/chexpert/CheXpert-v1.0-small/train')
configs.add_variable('kind_of_transformation_loss', 'l2')
configs.add_variable('create_prediction_output_file', False)
configs.add_variable('example_identifier_columns', ['scanid','subjectid','PFTExam_Global_ID','CRStudy_Global_ID', 'pftid', 'crstudy'])
configs.add_variable('limit_training_examples', False)
configs.add_variable('max_training_examples',1000)
configs.add_variable('unary_input_multiplier', -1)
configs.add_variable('load_dataset_to_memory', False)
configs.add_variable('meta_file_root', '/usr/sci/projects/DeepLearning/Tolga_Lab/Projects/Project_JoyceSchroeder/data/data_PFT/MetaData/MetaData_')
configs.add_variable('subtract_0.7', False)
configs.add_variable('use_smokeless_history', False)
configs.add_variable('prevent_horizontal_flip_in_lateral', True)
configs.add_variable('multiplier_lr_scheduler', 0.1)
configs.add_variable('revert_model_lr_scheduler', False)
configs.add_variable('use_amsgrad', False)
configs.add_variable('use_cut_restrictive', False)
configs.add_variable('two_inputs', True)
configs.add_variable('image_to_use_for_one_input_restrictive', 'zoom')

#These are the main configs to change from default
configs.add_variable('trainable_densenet', False)
configs.add_variable('use_conv11', False)
configs.add_variable('labels_to_use', 'only_absolute') # 'two_ratios', 'three_absolute', 'all_nine',
                                                       #'only_absolute','none', 'fev1fvc_predrug_absolute',
                                                       #'predict_diffs', 'fev1_ratio', 'fev1fvc_predrug'
                                                       # restrictive
configs.add_variable('use_lateral', False)
configs.add_variable('tie_cnns_same_weights', False)
configs.add_variable('tie_conv11_same_weights', False)

# configuration of architecture of end of model
configs.add_variable('n_hidden_layers', 2)
configs.add_variable('channels_hidden_layers', 2048) # only used if configs['n_hidden_layers']>0
configs.add_variable('use_dropout_hidden_layers', 0.0) # 0 turn off dropout; 0.25 gives seems to give about the same results as l = 0.05 # only used if configs['n_hidden_layers']>0
configs.add_variable('use_batchnormalization_hidden_layers', False)
configs.add_variable('conv11_channels', 128) # only used if configs['use_conv11']
configs.add_variable('network_output_kind', 'linear') #'linear', 'softplus' , 'sigmoid'
configs.add_variable('individual_output_kind',{'copd':'sigmoid'})

# these configs are modified depending on model and machine
configs.add_variable('machine_to_use', 'dgx' if socket.gethostname() == 'rigveda' 
                                        else 'titan' if (socket.gethostname() =='linux-55p6' or socket.gethostname() == 'titan')   
                                        else 'atlas' if (socket.gethostname() == 'atlas')   
                                        else 'other')
configs.add_variable('remove_pre_avg_pool', True)
configs.add_variable('l2_reg_fc', 0.05)
configs.add_variable('l2_reg_cnn', 0.0)
configs.add_variable('l2_reg_location', 0.0)

configs.add_variable('use_sigmoid_safety_constants',False)
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
                        'COPD':'copd',
                        'fev1_diff':'fev1_diff',
                        'fvc_diff':'fvc_diff',
                        'AGE_AT_PFT':'age',
                        'GENDER':'gender',
                        'TOBACCO_STATUS':'tobacco_status',
                        'SMOKING_TOBACCO_STATUS':'smoking_tobacco_status',
                        'LUNG_TRANSPLANT':'lung_transplant'})

configs.add_variable('all_input_columns',lambda self: list(self['columns_translations'].values()))
configs.add_variable('all_output_columns',['fvc_pred',
                        'fev1_pred',
                        'fev1fvc_pred',
                        'fvc_predrug',
                        'fev1_predrug',
                        'fev1fvc_predrug',
                        'fvc_ratio',
                        'fev1_ratio',
                        'fev1fvc_ratio',
                        'copd',
                        'fev1_diff',
                        'fvc_diff',
                        'gold'])


configs.add_variable('BATCH_SIZE',lambda self: get_batch_size(self))

configs.add_variable('get_individual_kind_of_loss',lambda self: get_individual_kind_of_loss(self))

configs.add_variable('get_individual_output_kind',lambda self: get_individual_output_kind(self))

configs.add_variable('get_individual_pre_transformation',lambda self: get_individual_pre_transformation(self))

configs.add_variable('get_individual_loss_weights',lambda self: get_individual_loss_weights(self))

# defining all variables that the network should output
configs.add_self_referenced_variable_from_dict('get_labels_columns_pft', 'labels_to_use',
                                      {'two_ratios': ['fev1fvc_predrug','fev1_ratio'],
                                       'three_absolute':['fev1_predrug','fvc_predrug', 'fev1_pred'],
                                       'all_nine':['fvc_pred','fev1_pred','fev1fvc_pred','fvc_predrug','fev1_predrug','fev1fvc_predrug','fvc_ratio','fev1_ratio','fev1fvc_ratio'],
                                       'only_absolute':['fev1_predrug','fvc_predrug', 'fev1_pred', 'fvc_pred'],
                                       'fev1fvc_predrug_absolute':['fev1_predrug','fvc_predrug'],
                                       'predict_diffs':['fev1_diff','fvc_diff'],
                                       'two_predrug_absolute':['fev1_predrug','fvc_predrug'],
                                       'fev1fvc_predrug':['fev1fvc_predrug'],
                                       'fev1_ratio':['fev1_ratio'],
                                       'restrictive':['fev1fvc_predrug','fvc_ratio'],
                                       'none':[]})

configs.add_self_referenced_variable_from_dict('avg_pool_kernel_size', 'chexnet_architecture',
                                      {'densenet': 7,
                                       'resnet':7,
                                       'vgg': 7,
                                       'inception':8,
                                       'squeezenet': 13 ,
                                       'alexnet': 6
                                       })

configs.add_self_referenced_variable_from_dict('get_labels_columns_copd', 'output_copd',
                                      {True: ['copd'], False: []})

configs.add_self_referenced_variable_from_dict('get_labels_columns_gold', 'output_gold',
                                      {True: ['gold'], False: []})

configs.add_variable('get_labels_columns',lambda self: self['get_labels_columns_pft'] + self['get_labels_columns_copd']+ self['get_labels_columns_gold'])

configs.add_self_referenced_variable_from_dict('pft_plot_columns', 'labels_to_use',
                                      {'two_ratios': [['fev1fvc_predrug'],['fev1_ratio']],
                                       'three_absolute':[['fev1_predrug','fvc_predrug', 'fev1_pred'],['fev1fvc_predrug'],['fev1_ratio']],
                                       'all_nine':[['fvc_pred','fev1_pred','fev1fvc_pred','fvc_predrug','fev1_predrug','fev1fvc_predrug','fvc_ratio','fev1_ratio','fev1fvc_ratio']],
                                       'only_absolute':[['fev1_predrug','fvc_predrug', 'fev1_pred', 'fvc_pred'], ['fev1fvc_predrug'],['fev1_ratio']],
                                       'fev1fvc_predrug_absolute':[['fev1_predrug','fvc_predrug'], ['fev1fvc_predrug']],
                                       'predict_diffs':[['fev1_diff','fvc_diff'], ['fev1fvc_predrug'], ['fev1_ratio']],
                                       'two_predrug_absolute':[['fev1_predrug','fvc_predrug'], ['fev1fvc_predrug'], ['fev1_ratio']],
                                       'fev1fvc_predrug':[['fev1fvc_predrug']],
                                       'fev1_ratio':[['fev1_ratio']],
                                       'restrictive': [['fev1fvc_predrug'],['fvc_ratio']],
                                       'none':[]})

configs.add_variable('pre_transform_labels', PreTransformLabels(configs))


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

configs.add_predefined_set_of_configs('fc20180524', {'use_true_predicted': True
                                                ,'use_lateral': True
                                                ,'kind_of_loss':'l1'
                                                ,'network_output_kind':'softplus'
                                                ,'labels_to_use':'two_ratios'
                                                ,'use_extra_inputs': True
                                                ,'use_batchnormalization_hidden_layers': True
                                                ,'use_random_crops': True
                                                ,'positions_to_use': ['PA', 'AP']
                                                ,'dropout_batch_normalization_last_layer':True
                                                ,'l2_reg_fc': 0.0
                                                ,'initial_lr_cnn': 1e-04
                                                ,'use_dropout_hidden_layers': 0.25
                                                ,'initial_lr_fc': 0.0001})

configs.add_predefined_set_of_configs('resnet18', {'chexnet_architecture': 'resnet'
                                                ,'chexnet_layers': 18
                                                ,'CKPT_PATH':'model_chestxray14_resnet_18.t7'})

configs.add_predefined_set_of_configs('cnn20180628', {'use_true_predicted': True
                                                ,'use_lateral': True
                                                ,'kind_of_loss':'l1'
                                                ,'network_output_kind':'softplus'
                                                ,'labels_to_use':'two_ratios'
                                                , 'trainable_densenet': True
                                                ,'use_extra_inputs': True
                                                ,'use_batchnormalization_hidden_layers': True
                                                ,'use_random_crops': True
                                                ,'positions_to_use': ['PA', 'AP']
                                                ,'dropout_batch_normalization_last_layer':True
                                                 , 'densenet_dropout':0.25
                                                ,'l2_reg_fc': 0.0
                                                ,'initial_lr_cnn': 1e-04
                                                ,'use_dropout_hidden_layers': 0.25
                                                ,'initial_lr_fc': 0.0001
                                                ,'BATCH_SIZE':22})

configs.add_predefined_set_of_configs('meanvar_loss', {'use_mean_var_loss': True
                                                ,'kind_of_loss': 'l2'
                                                #,'use_dropout_hidden_layers':0.0
                                                ,'dropout_batch_normalization_last_layer':False
                                                , 'network_output_kind': 'linear'
                                                ,'initial_lr_fc': 0.0001
                                                ,'initial_lr_cnn': 0.0001
                                                ,'BATCH_SIZE': 64})

configs.add_predefined_set_of_configs('vrgan_train', { 'use_lateral': False,
                                             'use_horizontal_flip': False,
                                             'use_random_crops': True,
                                             'histogram_equalization': 'global',
                                             'use_extra_inputs': False,
                                             'individual_output_kind': {'copd':'linear'},
                                             'unary_input_multiplier': 1,
                                             'remove_cases_more_one_image_per_position': False,
                                             'maximum_date_diff':30,
                                             'load_dataset_to_memory': True,
                                             'network_output_kind': 'linear',
                                             'n_hidden_layers': 0,
                                             'use_dropout_hidden_layers': 0.0
                                             })

configs.add_predefined_set_of_configs('vrgan_eval', { 'use_lateral': False,
                                             'use_horizontal_flip': False,
                                             'use_random_crops': True,
                                             'histogram_equalization': 'global',
                                             'use_extra_inputs': False,
                                             'individual_output_kind': {'copd':'linear'},
                                             'unary_input_multiplier': 1,
                                             'remove_cases_more_one_image_per_position': False,
                                             'maximum_date_diff':30,
                                             'magnification_input': 4,
                                             'network_output_kind': 'linear',
                                             'n_hidden_layers': 0,
                                             'use_dropout_hidden_layers': 0.0
                                             })
                                             
configs.add_self_referenced_variable_from_dict('get_available_memory', 'machine_to_use',
                                      {'dgx': 15600-550-10,
                                       'titan':11700-2978,#11700-550-10,
                                       'atlas':23000-2000,
                                       'other':9000-2*550-600})
def get_batch_size(self):
    if self['trainable_densenet']:
        return int(self['get_available_memory']/140./(2. if self['use_lateral'] else 1))
    else:
        return 128

def get_individual_characteristic(individual_string, general_string, self):
    #return [(self[individual_string][self['get_labels_columns'][k]] if (self['get_labels_columns'][k] in list(self[individual_string].keys())) else self[general_string]) for k in range(len(self['get_labels_columns']))]
    return {self['all_output_columns'][k]:(self[individual_string][self['all_output_columns'][k]] if (self['all_output_columns'][k] in list(self[individual_string].keys())) else self[general_string]) for k in range(len(self['all_output_columns']))}

def get_individual_kind_of_loss(self):
    return get_individual_characteristic('individual_kind_of_loss', 'kind_of_loss', self)

def get_individual_output_kind(self):
    return get_individual_characteristic('individual_output_kind', 'network_output_kind', self)

def get_individual_pre_transformation(self):
    return get_individual_characteristic('individual_pre_transformation', 'pre_transformation', self)

def get_individual_loss_weights(self):
    return get_individual_characteristic('individual_loss_weights', 'loss_weight', self)


configs.add_predefined_set_of_configs('frozen_densenet', {})

configs.add_predefined_set_of_configs('copd_only', {'kind_of_loss':'bce',
                                                          'labels_to_use':'none',
                                                          'network_output_kind':'sigmoid',
                                                          'output_copd':True})
