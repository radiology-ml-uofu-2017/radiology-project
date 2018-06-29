from future.utils import raise_with_traceback
import re
import torch
import torch.nn as nn
from configs import configs
import torchvision
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import time
import logging 
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import os
from sklearn.decomposition import PCA
import h5py
from collections import Counter

np.seterr(divide = "raise", over="warn", under="warn",  invalid="raise")

labels_columns = configs['get_labels_columns']

def cmp(a, b):
    return (a > b) - (a < b)
  
def compare_versions(version1, version2):
    def normalize(v):
        return [int(x) for x in re.sub(r'(\.0+)*$','', v).split(".")]
    return cmp(normalize(version1), normalize(version2))
  
def load_checkpoint(model):
    if configs['CKPT_PATH'] is not None:
        checkpoint = torch.load(configs['CKPT_PATH'])
        if configs['CKPT_PATH'].endswith('.tar'):
            state_dict = checkpoint['state_dict']
        elif configs['CKPT_PATH'].endswith('.t7'):
            state_dict = checkpoint
        
        #block taken from torchvision 0.2.1 source code. Necessary since 
        # in pytroch 0.4.0 modules could not ahve "." in their name 
        # anymore
        if configs['CKPT_PATH']=='densenet121.pth.tar':
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                new_key = key
                if res and compare_versions(torchvision.__version__, '0.2.1')>=0:
                    new_key = res.group(1) + res.group(2)
                new_key = new_key.replace('densenet121', 'model')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model

def load_pretrained_chexnet():
    chexnetModel = CheXNet(14, configs['chexnet_layers'], configs['chexnet_architecture']).cuda()
    chexnetModel = torch.nn.DataParallel(chexnetModel).cuda()
    chexnetModel = load_checkpoint(chexnetModel)
    return chexnetModel

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def get_model_cnn():
    x = load_pretrained_chexnet()
    x.module.set_classifier_containing_avg_pool_part(nn.Sequential())
    return x.module.cuda()

def get_qt_inputs():
    n_inputs = 1
    if configs['use_lateral']:
        n_inputs = 2
    if configs['tie_cnns_same_weights']:
        n_cnns = 1
    else:
        n_cnns = n_inputs 
    return n_inputs, n_cnns

class FinalLayers(nn.Module):
    def __init__(self, num_ftrs, n_inputs):
        super(FinalLayers, self).__init__()
        self.n_inputs = n_inputs
        self.spatial_part = ModelSpatialToFlatPart(num_ftrs*self.n_inputs)
        if not configs['use_dsnm']:
            self.fc_part = ModelFCPart(self.spatial_part.current_n_channels)
        else:
            self.fc_part = ModelDSNMPart(self.spatial_part.current_n_channels) 
        self.final_linear = ModelLastLinearLayer(self.fc_part.current_n_channels)
    
    def forward(self, x, extra_fc_input ):
        x = self.spatial_part(x)
        x = self.fc_part(x, extra_fc_input)

        return self.final_linear(x)
      
class ModelMoreInputs(nn.Module):
    def __init__(self, num_ftrs):
        super(ModelMoreInputs, self).__init__()
        
        self.n_inputs, self.n_cnns = get_qt_inputs()
        
        cnns = []
        for i in range(self.n_cnns):
            if configs['trainable_densenet']:
                new_cnn = get_model_cnn()
                num_ftrs = new_cnn.num_ftrs
                cnns.append(new_cnn)
            else:
                cnns.append(nn.Sequential())
        self.cnn = nn.ModuleList(cnns)
        self.final_layers = FinalLayers(num_ftrs, self.n_inputs)
        
    def forward(self, images, extra_fc_input ):
        all_outputs = []
        if not images.size()[1]==self.n_inputs:
            raise_with_traceback(ValueError('Wrong number of arguments for the model forward function. Expected ' + str(self.n_inputs) + ' and received ' + str(args.size()[1])))
        for i in range(images.size()[1]):
            if configs['tie_cnns_same_weights']:
                index = 0
            else:
                index = i
            all_outputs.append(self.cnn[index](images[:,i,...]))
        x, probs, averages = self.final_layers(all_outputs, extra_fc_input)

        return x, probs, averages

def get_spatial_part_fc(num_ftrs):
    spatial_part = torch.nn.Sequential()
    spatial_part.add_module("spatial_output",torch.nn.Sequential())
    current_n_channels = num_ftrs
    if configs['use_conv11']:
        if configs['use_batchnormalization_hidden_layers']:
            spatial_part.add_module("bn_conv11",torch.nn.BatchNorm2d(current_n_channels).cuda())
        spatial_part.add_module("conv11",nn.Conv2d( in_channels = current_n_channels, out_channels =  configs['conv11_channels'], kernel_size = 1).cuda())
        spatial_part.add_module("reluconv11",nn.ReLU().cuda())
        current_n_channels = configs['conv11_channels']
    if not configs['remove_pre_avg_pool']:
        spatial_part.add_module("avgpool",nn.AvgPool2d(7).cuda())
    else:
        current_n_channels = current_n_channels*49
    
    spatial_part.add_module("flatten",Flatten().cuda())
    return spatial_part, current_n_channels

class MeanVarLossMean(nn.Module):
    def __init__(self, col_name, bins):
        super(MeanVarLossMean, self).__init__()
        self.bins = bins
        
    def forward(self, input):
        x = torch.sum(input*self.bins.expand_as(input), dim = 1).squeeze(1)
        return x

class MeanVarLossOutput(nn.Module):
    def __init__(self, in_features, bins):
        super(MeanVarLossOutput, self).__init__()
        self.bins = bins
        intermediary_probabilities_list = []
        outputs_list = []
        for index, output_column_name in enumerate(configs['get_labels_columns']):
            output_size_linear_layer= bins[index].size()[1]
            one_output = nn.Sequential()
            one_output.add_module('linear_to_logits', nn.Linear(in_features, output_size_linear_layer))
            
            intermediary_probabilities_list.append(one_output)
            one_output = nn.Sequential()
            one_output.add_module('logits_to_probs', nn.Softmax())
            one_output.add_module('probs_to_mean', MeanVarLossMean(output_column_name, bins[index]))
            outputs_list.append(one_output)
        self.intermediary_probabilities_module_list = nn.ModuleList(intermediary_probabilities_list)
        self.outputs_module_list = nn.ModuleList(outputs_list)

    def forward(self, input):
        intermediary_logits_list = []
        outputs_list = []
        size0 = input.size()[0]
        argmax_list = []
        for index, output_column_name in enumerate(configs['get_labels_columns']):
            intermediary_logits_one_output = self.intermediary_probabilities_module_list[index](input)
            intermediary_logits_list.append(intermediary_logits_one_output)
            one_output = self.outputs_module_list[index](intermediary_logits_one_output)
            outputs_list.append(one_output)
            argmax_list.append(torch.index_select(self.bins[index], dim = 1, index = torch.max(intermediary_logits_one_output, dim=1)[1].squeeze(1)).squeeze(0))
        distribution_averages = torch.stack(outputs_list,dim = 1)
        outputs = distribution_averages
        return outputs, intermediary_logits_list, distribution_averages

class MultiplyInGroups(nn.Module):
    def __init__(self, n_groups):
        super(MultiplyInGroups, self).__init__()
        self.n_groups = n_groups
    
    def forward(self, input):
        return multiply_in_groups(input, self.n_groups)
    
def multiply_in_groups(input, n_groups):
    original_shape = input.size()
    x = input.view(original_shape[0],n_groups,-1)
    x = torch.sigmoid(x)
    x = torch.prod(x, dim = 2)

    x = x.view(original_shape[0], n_groups)
    return x

class PCAForward(nn.Module):
    def __init__(self, pca):
        super(PCAForward, self).__init__()
        
        self.weight = torch.nn.Parameter(torch.from_numpy(pca.components_).cuda())
        self.bias = torch.nn.Parameter(torch.from_numpy(np.expand_dims(pca.mean_,axis = 0)).cuda())
        
        
    def forward(self, x):
        x -= self.bias.expand(x.size(0), self.bias.size(1))
        x = self._backend.Linear()(x, self.weight)
        return x

class ModelDSNMPart(nn.Module):
    def __init__(self, in_features):
        super(ModelDSNMPart, self).__init__()
        self.current_n_channels = in_features
        self.n_groups = configs['dsnm_n_groups']
        if configs['use_pca_dsnm']:
            linear_input_features = configs['dsnm_pca_size']
        else:   
            linear_input_features = self.current_n_channels
        self.dsnm = nn.Sequential(nn.Linear(linear_input_features, self.n_groups*(self.n_groups-1)),MultiplyInGroups(self.n_groups))        
        self.current_n_channels = self.n_groups
        
        self.initialized = False
        self.acumulated_x = None
        
    def initialize(self):
        assert(not self.initialized)
        preprocessed_images = self.acumulated_x
        
        if configs['use_pca_dsnm']:
            if configs['dsnm_pca_size']>self.acumulated_x.shape[0]:
                raise_with_traceback(ValueError("configs['dsnm_pca_size'] ("+configs['dsnm_pca_size']+") is set to bigger than the number of training images ("+self.acumulated_x.size(0) +")"))
            self.pca = PCA(n_components=configs['dsnm_pca_size'])
            self.pca.fit(preprocessed_images)
            pcaed_images = self.pca.transform(preprocessed_images)
            self.pca_forward = PCAForward(self.pca)
        else:
            pcaed_images = preprocessed_images
         
        w,b, self.kmeans = get_unsupervised_weights(pcaed_images, self.n_groups, 'dsnmkmeansout')
        
        self.dsnm.apply(lambda tensor: init_k_means(tensor, w, b))
        
        self.initialized = True
        
    def forward(self, x, extra_fc_input = None):
        if not self.initialized:
            if self.acumulated_x is None:
                self.acumulated_x = x.data.cpu().numpy()
            else:
                self.acumulated_x = np.concatenate((self.acumulated_x, x.data.cpu().numpy()), axis=0)
            return (x*0)[:,0:self.current_n_channels]
        if configs['use_pca_dsnm']:
            x = self.pca_forward(x)

        x = self.dsnm(x)

        return x
    
class ModelSpatialToFlatPart(nn.Module):
    def __init__(self, num_ftrs):
        super(ModelSpatialToFlatPart, self).__init__()
        
        _, self.n_cnns = get_qt_inputs()
        spatial_parts = []
        for i in range(self.n_cnns):
            this_spatial_part, self.current_n_channels = get_spatial_part_fc(num_ftrs)
            spatial_parts.append(this_spatial_part)
        self.spatial_part = nn.ModuleList(spatial_parts)
        self.spatial_part.apply(weights_init)
        
    def forward(self, spatial_outputs):
        all_outputs = []
        for i, spatial_output in enumerate(spatial_outputs):
            if configs['tie_conv11_same_weights']:
                index = 0
            else:
                index = i            
            all_outputs.append(self.spatial_part[index](spatial_output.contiguous()))
        x = torch.cat(all_outputs, 1)
        return x
      
class ModelLastLinearLayer(nn.Module):
    def __init__(self, current_n_channels):
        super(ModelLastLinearLayer, self).__init__()
        self.final_linear_layer = torch.nn.Sequential()
        self.current_n_channels = current_n_channels
        if configs['use_mean_var_loss']:
            self.bins_list = []
            for index, col_name in enumerate(configs['get_labels_columns']):
                if configs['use_sigmoid_safety_constants']:
                    min_range = configs['pre_transform_labels'].ranges_labels[col_name][0]*configs['sigmoid_safety_constant'][col_name][0]
                    max_range = configs['pre_transform_labels'].ranges_labels[col_name][1]*configs['sigmoid_safety_constant'][col_name][1]
                else:
                    min_range = configs['pre_transform_labels'].ranges_labels[col_name][0]
                    max_range = configs['pre_transform_labels'].ranges_labels[col_name][1]
                spacing = configs['meanvarloss_discretization_spacing']
                self.bins_list.append(torch.autograd.Variable(torch.arange(min_range, max_range+spacing/2.0, spacing).unsqueeze(0).cuda(async=True, device = 0), requires_grad=False) )
                
            linear_out_model = MeanVarLossOutput(self.current_n_channels, self.bins_list).cuda()
        else:
            linear_out_model = nn.Linear(self.current_n_channels , len(labels_columns)).cuda()
        
        self.final_linear_layer.add_module("linear_out", linear_out_model)
        self.final_linear_layer.apply(weights_init)
    def forward(self, input):
        if configs['use_mean_var_loss']:
            x, logits, averages = self.final_linear_layer(input)
        else:
            x = self.final_linear_layer(input)
            logits = None
            averages = None
        output_kind_each_output = [ configs['get_individual_output_kind'][name] for name in configs['get_labels_columns']]
        dic_output_kinds = {'linear':nn.Sequential(),'softplus':nn.Sequential(nn.Softplus().cuda()), 'sigmoid':nn.Sequential(nn.Sigmoid().cuda())}
        #add exception when output_kind_each_output cotains element not in dic_output_kinds.keys()
        unrecognized_kinds_of_outputs = list(set(output_kind_each_output).difference( dic_output_kinds.keys()) )
        if len(unrecognized_kinds_of_outputs)>0:
            raise_with_traceback(ValueError('There are output kinds in configs["individual_output_kind"] or configs["network_output_kind"] that are not one of: linear, sigmoid and softplus: ' + str(unrecognized_kinds_of_outputs)))
        all_masked_outputs = []
        for output_kind in list(dic_output_kinds.keys()):
            mask = torch.autograd.Variable(torch.FloatTensor(np.repeat(np.expand_dims([(1.0 if output_kind_each_output[k] == output_kind else 0.0) for k in range(len(output_kind_each_output))], axis = 0), dic_output_kinds[output_kind](x).size()[0], axis=0)).cuda(), volatile = False)
            all_masked_outputs.append((dic_output_kinds[output_kind](x)*mask).unsqueeze(2))
        x = torch.cat(all_masked_outputs, 2)
        x = torch.sum(x, 2).squeeze(2)
        return x, logits, averages
      
      
class ModelFCPart(nn.Module):
    def __init__(self, current_n_channels):
        super(ModelFCPart, self).__init__()
        self.current_n_channels = current_n_channels
        activation_function_dict = {'relu': nn.ReLU().cuda(), 'tanh':nn.Tanh().cuda(), 
                            'sigmoid':nn.Sigmoid().cuda(), 'softplus':nn.Softplus().cuda()
        }
    
        activation_function = activation_function_dict[configs['fc_activation']]
    
        self.fc_before_extra_inputs = torch.nn.Sequential()
        self.fc_after_extra_inputs = torch.nn.Sequential()
        
        current_model = self.fc_before_extra_inputs
        for layer_i in range(configs['n_hidden_layers']): 
            if (configs['layer_to_insert_extra_inputs']== layer_i) and configs['use_extra_inputs']:
                self.current_n_channels = self.current_n_channels+15
                current_model = self.fc_after_extra_inputs
            if configs['use_batchnormalization_hidden_layers']:
                current_model.add_module("bn_"+str(layer_i),torch.nn.BatchNorm1d(self.current_n_channels).cuda())
            current_model.add_module("drop_"+str(layer_i),nn.Dropout(p=configs['use_dropout_hidden_layers']).cuda())
            # On DGX, this next line, in the first iteration of the loop, is the one taking a long time (about 250s) in the .cuda() part.
            # probably because of some incompatibility between Cuda 9.0 and pytorch 0.1.12
            current_model.add_module("linear_"+str(layer_i),nn.Linear(self.current_n_channels, configs['channels_hidden_layers'] ).cuda()) 
            current_model.add_module("nonlinearity_"+str(layer_i),activation_function)
            self.current_n_channels = configs['channels_hidden_layers']
        
        if configs['use_extra_inputs'] and (configs['layer_to_insert_extra_inputs']==(configs['n_hidden_layers']+1)):
            self.current_n_channels = self.current_n_channels+15
            current_model = self.fc_after_extra_inputs
        
        if configs['use_extra_inputs'] and configs['layer_to_insert_extra_inputs']>(configs['n_hidden_layers']+1):
            raise_with_traceback(ValueError("configs['layer_to_insert_extra_inputs'] ("+configs['layer_to_insert_extra_inputs']+") is set to bigger than configs['n_hidden_layers']+1 ("+configs['n_hidden_layers']+1 +")"))
        if configs['dropout_batch_normalization_last_layer']:
            if configs['use_batchnormalization_hidden_layers']:
                current_model.add_module("bn_out",torch.nn.BatchNorm1d(self.current_n_channels).cuda())
            current_model.add_module("drop_out",nn.Dropout(p=configs['use_dropout_hidden_layers']).cuda())
                                 
        self.fc_after_extra_inputs.apply(weights_init)
        self.fc_before_extra_inputs.apply(weights_init)
      
    def forward(self, input, extra_fc_input = None):
      
        x = self.fc_before_extra_inputs(input)
        if extra_fc_input is not None:
            x = torch.cat((extra_fc_input,x),1)
        x = self.fc_after_extra_inputs(x)
        return x

class NonDataParallel(nn.Module):
    def __init__(self, model):
        super(NonDataParallel, self).__init__()
        self.module = model
    
    def forward(self, input, extra_inputs):
        return self.module(input, extra_inputs)

def get_model(num_ftrs):
    outmodel = ModelMoreInputs(num_ftrs)
    if configs['use_more_one_gpu']:
        outmodel = torch.nn.DataParallel(outmodel).cuda()
    else:
        outmodel = NonDataParallel(outmodel).cuda()
    return outmodel

class Reference:
    def __init__(self):
        pass

    def get(self):
        return self._value

    def set_variable(self, val):
        self._value = val
        
    def set_value(self, val):
        self._value = val

class CheXNet(nn.Module):
    def __init__(self, out_size, num_layers = 121, architecture = 'densenet'):
        super(CheXNet, self).__init__()
        self.architecture = architecture
        model_parameters = {'pretrained':False}
        if architecture=='densenet':
            model_parameters['drop_rate'] = configs['densenet_dropout']
            num_layers_to_model = {121:torchvision.models.densenet121, 169:torchvision.models.densenet169, 201:torchvision.models.densenet201, 161:torchvision.models.densenet161}
        elif architecture=='resnet':
            num_layers_to_model = {18:torchvision.models.resnet18, 34:torchvision.models.resnet34, 50:torchvision.models.resnet50, 101:torchvision.models.resnet101, 152:torchvision.models.resnet152}
        
        self.model = num_layers_to_model[num_layers](**model_parameters)

        self.num_ftrs = self.get_classifier().in_features
        new_last_layer = nn.Sequential(
            nn.Linear(self.num_ftrs, out_size),
            nn.Sigmoid()
        )
        
        self._set_classifier(new_last_layer)
            
        self.modified_end_avg_pool = False
    
    def get_classifier(self):
        if self.architecture=='densenet':
            return self.model.classifier
        elif self.architecture=='resnet':
            return self.model.fc
        
    def _set_classifier(self, new_last_layer):
        if self.architecture=='densenet':
            self.model.classifier = new_last_layer
        elif self.architecture=='resnet':
            self.model.fc = new_last_layer
    
    def set_classifier_containing_avg_pool_part(self, classifier):
        self._set_classifier(classifier)
        #self.model.classifier = classifier
        self.modified_end_avg_pool = True
        
    def forward(self, x):
        if self.modified_end_avg_pool:
            if self.architecture=='resnet':
                x = self.model.conv1(x)
                x = self.model.bn1(x)
                x = self.model.relu(x)
                x = self.model.maxpool(x)

                x = self.model.layer1(x)
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                x = self.model.layer4(x)
            elif self.architecture=='densenet':
                x = self.model.features(x)
                x = F.relu(x, inplace=True) # should I always have this relu on?
            
            x = self.get_classifier()(x)
        else:
            x = self.model(x)
        return x   

'''
#another way of integrating several different models
class MyStandardizedTorchModel(object):
    def __init__(self, model):
        self.model = model
      
    def forward(self, x):
        self.features(x)
        self.classifier(x)
        return x
      
class MyResNet(MyStandardizedTorchModel):
    def __init__(self, model):
        super(MyResNet, self).__init__(model)
        self.classifier_in_features = model.fc.in_features
        
    def classifier(self,x):
        return self.model.fc(x)
        
    def features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x

class MyDenseNet(MyStandardizedTorchModel):
    def __init__(self, model):
        super(MyDenseNet, self).__init__(model)
        self.classifier_in_features = model.classifier.in_features

    def classifier(self,x):
        return self.model.classifier(x)
        
    def features(self, x):
        x = self.model.features(x)
        x = F.relu(x, inplace=True) # should I always have this relu on?
        return x
      
class CheXNet(nn.Module):
    def __init__(self, out_size, num_layers = 121, architecture = 'densenet'):
        super(CheXNet, self).__init__()
        self.architecture = architecture
        model_parameters = {'pretrained':False}
        if architecture=='densenet':
            model_parameters['drop_rate'] = configs['densenet_dropout']
            num_layers_to_model = {121:torchvision.models.densenet121, 169:torchvision.models.densenet169, 201:torchvision.models.densenet201, 161:torchvision.models.densenet161}
        elif architecture=='resnet':
            num_layers_to_model = {18:torchvision.models.resnet18, 34:torchvision.models.resnet34, 50:torchvision.models.resnet50, 101:torchvision.models.resnet101, 152:torchvision.models.resnet152}
        
        self.model = {'densenet':MyDenseNet, 'resnet':MyResNet}[architecture](num_layers_to_model[num_layers](**model_parameters))
        #self.model = torchvision.models.densenet121(pretrained=False, drop_rate = configs['densenet_dropout'])
        
        num_ftrs = self.model.classifier_in_features
        
        self.model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )
        
        self.modified_end_avg_pool = False
        
    def set_classifier_containing_avg_pool_part(self, classifier):
        self.model.classifier = classifier
        self.modified_end_avg_pool = True
        
    def forward(self, x):
        x = self.model.features(x)
        if not self.modified_end_avg_pool:
            x = F.avg_pool2d(x, kernel_size=7, stride=1).view(x.size(0), -1)
        x = self.model.classifier(x)
        return x   
'''

def get_unsupervised_weights(inputs, nClusters, filename):
    save_and_load_wb = False
    if save_and_load_wb:
        if os.path.isfile('w'+filename):
            h5f = h5py.File('w'+filename,'r')
            w = h5f['dataset_1'][:]
                
            h5f.close()
            
            h5f = h5py.File('b'+filename,'r')
            b = h5f['dataset_1'][:]
            h5f.close()
            return w, b
    
    totalWeights = (nClusters * (nClusters - 1))
    N_CHANNELS = inputs.shape[1]
    kmeans = KMeans(n_clusters=nClusters, init='k-means++').fit(inputs)
    print(Counter(kmeans.labels_))
    cc = kmeans.cluster_centers_
    clusterCenter = np.transpose(cc)

    # intialize the weight matrix and bias matrix     
    weight_matrix = np.zeros((N_CHANNELS,totalWeights),dtype=np.float32)
    b_matrix = np.zeros((1,totalWeights),dtype=np.float32)
    currentCluster = np.zeros((N_CHANNELS,1),dtype=np.float32)
    distances = np.zeros((totalWeights),dtype=np.float32)

    ctr = 0
    for k in range(nClusters):
        compareWeightindex = range(0,k)+range(k+1, nClusters)
        currentCluster[:,0] = clusterCenter[:,k]
        repcurrentCluster = currentCluster.repeat(len(compareWeightindex),axis=1)
        diff = repcurrentCluster - clusterCenter[:,compareWeightindex] 
        absdiff = np.linalg.norm(diff, axis = 0)
        v = diff/absdiff
        v[np.isnan(v)]=0 
        v[np.isinf(v)]=0 
        # bias
        b = -np.sum(v * ( 0.5*(repcurrentCluster + clusterCenter[:,compareWeightindex])),axis=0)
        # update the weights and bias
        weight_matrix[:,ctr:ctr+len(compareWeightindex)]=v
        distances[ctr:ctr+len(compareWeightindex)] = np.sum(np.square(diff),axis = 0)
        b_matrix[:,ctr:ctr+len(compareWeightindex)]=b
        ctr = ctr + len(compareWeightindex)
    w = weight_matrix.reshape(N_CHANNELS, totalWeights) 
    b = b_matrix.reshape(totalWeights)
    
    if save_and_load_wb:
        h5f = h5py.File('w'+filename, 'w')
        h5f.create_dataset('dataset_1', data=w)
        h5f.close()
        h5f = h5py.File('b'+filename, 'w')
        h5f.create_dataset('dataset_1', data=b)
        h5f.close()
        
    return w, b

def init_k_means(tensor, w, b):
    if hasattr(tensor, 'weight') or hasattr(tensor, 'bias'):
        if isinstance(tensor, torch.autograd.Variable):
            constant(tensor.data, val)
            return tensor
        if not tensor.weight.size()==torch.nn.Parameter(torch.from_numpy(w).permute(1,0)).size():
            raise_with_traceback(ValueError('Internal Bug Found: dimension of tensor.weight('+str(tensor.weight.size())+') and torch.nn.Parameter(torch.from_numpy(w).permute(1,0)) ('+str(torch.nn.Parameter(torch.from_numpy(w).permute(1,0)).size())+') are incompatible'))
        assert(tensor.bias.size()==torch.nn.Parameter(torch.from_numpy(b)).size())
        tensor.weight = torch.nn.Parameter(torch.from_numpy(w).permute(1,0).cuda())
        tensor.bias = torch.nn.Parameter(torch.from_numpy(b).cuda())
    return tensor
    
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if configs['weight_initialization'] == 'xavier':
            torch.nn.init.xavier_normal(m.weight, gain = torch.nn.init.calculate_gain('relu'))
        elif configs['weight_initialization'] == 'original':
            pass
        else:
            raise_with_traceback(ValueError('configs["weight_initialization"] was set to an invalid value: ' + configs["weight_initialization"]))
        if configs['bias_initialization'] == 'constant':
            torch.nn.init.constant(m.bias, 0.1)
        elif configs['bias_initialization'] == 'original':
            pass
        else:
            raise_with_traceback(ValueError('configs["bias_initialization"] was set to an invalid value: ' + configs["bias_initialization"]))

