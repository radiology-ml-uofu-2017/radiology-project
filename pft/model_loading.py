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
        state_dict = checkpoint['state_dict']
        
        #block taken from torchvision 0.2.1 source code. Necessary since 
        # in pytroch 0.4.0 modules could not ahve "." in their name 
        # anymore
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
    return model

def load_pretrained_chexnet():
    chexnetModel = DenseNet121(14).cuda()
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
    
class ModelMoreInputs(nn.Module):
    def __init__(self, num_ftrs):
        super(ModelMoreInputs, self).__init__()
        self.n_inputs = 1
        if configs['use_lateral']:
            self.n_inputs = 2
        self.fc = get_model_fc(num_ftrs*self.n_inputs)
        #self.cnn = get_model_cnn()
        
        cnns = []
        for i in range(self.n_inputs):
            if configs['trainable_densenet']:
                cnns.append(get_model_cnn())
            else:
                self.cnn.append(nn.Sequential())
        self.cnn = nn.ModuleList(cnns)
        
    def forward(self, args):
        #args = args.transpose(0,1).contiguous()
        #print(args.size())
        all_outputs = []
        if not args.size()[1]==self.n_inputs:
            raise_with_traceback(ValueError('Wrong number of arguments for the model forward function. Expected ' + str(self.n_inputs) + ' and received ' + str(args.size()[1])))
        #for i, input_ in enumerate(args):
        for i in range(args.size()[1]):
            pass
            #all_outputs.append(self.cnn[i](input_))
            all_outputs.append(self.cnn[i](args[:,i,...]))
        '''
        print(torch.cat(all_outputs, 0).size())
        print(torch.cat(all_outputs, 1).size())
        print(torch.cat(all_outputs, 2).size())
        print(torch.cat(all_outputs, 3).size())
        '''
        x = torch.cat(all_outputs, 1)
        x = self.fc(x)
        return x

def get_model_fc(num_ftrs):
    model = torch.nn.Sequential()
    current_n_channels = num_ftrs
    
    if configs['use_conv11']:
        if configs['use_batchnormalization_hidden_layers']:
            model.add_module("bn_conv11",torch.nn.BatchNorm2d(current_n_channels).cuda())
        model.add_module("conv11",nn.Conv2d( in_channels = current_n_channels, out_channels =  configs['conv11_channels'], kernel_size = 1).cuda())
        model.add_module("reluconv11",nn.ReLU().cuda())
        current_n_channels = configs['conv11_channels']
        
    if not configs['remove_pre_avg_pool']:
        model.add_module("avgpool",nn.AvgPool2d(7).cuda())
    else:
        current_n_channels = current_n_channels*49
    
    model.add_module("flatten",Flatten().cuda())
    
    for layer_i in range(configs['n_hidden_layers']): 
        model.add_module("drop_"+str(layer_i),nn.Dropout(p=configs['use_dropout_hidden_layers']).cuda())
        if configs['use_batchnormalization_hidden_layers']:
            model.add_module("bn_"+str(layer_i),torch.nn.BatchNorm1d(current_n_channels).cuda())
            
        # On DGX, this next line, in the first iteration of the loop, is the one taking a long time (about 250s) in the .cuda() part.
        # probably because of some incompatibility between Cuda 9.0 and pytorch 0.1.12
        model.add_module("linear_"+str(layer_i),nn.Linear(current_n_channels, configs['channels_hidden_layers'] ).cuda()) 
        model.add_module("relu_"+str(layer_i),nn.ReLU().cuda())
        current_n_channels = configs['channels_hidden_layers']
          
    model.add_module("linear_out", nn.Linear(current_n_channels , len(labels_columns)).cuda())
    
    if configs['network_output_kind']=='linear':
        pass
    elif configs['network_output_kind']=='softplus':
        model.add_module("softplus_out", nn.Softplus().cuda())
    elif configs['network_output_kind']=='sigmoid':
        model.add_module("sigmoid_out", nn.Sigmoid().cuda())
    else:
        raise_with_traceback(ValueError('configs["network_output_kind"] was set to an invalid value: ' + str(configs['network_output_kind'])))
    
    model.apply(weights_init)
    return model

def get_model(num_ftrs):
    outmodel = ModelMoreInputs(num_ftrs)
    outmodel = torch.nn.DataParallel(outmodel).cuda()
    #print(list(outmodel.parameters()))
    #model = get_model_fc(num_ftrs)
    #outmodel = load_pretrained_chexnet()
    #outmodel.module.set_classifier_containing_avg_pool_part(model)
    #outmodel = torch.nn.DataParallel(outmodel).cuda()
    return outmodel

class DenseNet121(nn.Module):
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
        self.modified_end_avg_pool = False
        
    def set_classifier_containing_avg_pool_part(self, classifier):
        self.densenet121.classifier = classifier
        self.modified_end_avg_pool = True
        
    def forward(self, x):
        if self.modified_end_avg_pool:
            x = self.densenet121.features(x)
            x = F.relu(x, inplace=True) # should I always have this relu on?
            x = self.densenet121.classifier(x)
        else:
            x = self.densenet121(x)
        return x   

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

