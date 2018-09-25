import os
from future.utils import raise_with_traceback
import pandas as pd
from configs import configs
from torch.utils.data import Dataset
import torch
from h5df import Store
import h5py
import torchvision.transforms as transforms
import image_preprocessing
import numpy as np
import metrics
from PIL import Image
import model_loading

try:
    import cPickle as pickle
except:
    import _pickle as pickle 


percentage_labels = configs['percentage_labels']

labels_columns = configs['get_labels_columns']

def write_filenames():
    pathFileTrain = '/home/sci/ricbl/Documents/projects/radiology-project/pft/train2.txt'
    pathPngs = '/usr/sci/projects/DeepLearning/Tolga_Lab/Projects/Project_JoyceSchroeder/data/data_PFT/'
    command = 'find ' + pathPngs + ' -type f -name "*.png" > ' + pathFileTrain
    os.system(command)

def read_filename(line):
    thisimage = {}

    lineItems = line.split()
    thisimage['filepath'] = lineItems[0]
    splitted_filepath = thisimage['filepath'].replace('\\','/').split("/")
    if splitted_filepath[-1] in [ 'PFT_000263_CRStudy_15_ScanID_1-W_Chest_Lat.png', 'PFT_000264_CRStudy_05_ScanID_2-W_Chest_Lat.png']:
        return None
    splitted_ids = splitted_filepath[-1].replace('-','_').replace('.','_').split('_')
    thisimage['subjectid'] = int(splitted_ids[1])
    thisimage['crstudy'] = int(splitted_ids[3])
    try:
        thisimage['scanid'] = int(splitted_ids[5])
    except ValueError:
        return None
    position = splitted_ids[-2].upper()
    if 'LARGE' in position:
        if configs['use_images_with_position_LARGE']:
            position = splitted_ids[-3].upper()
        else:
            return None
    if 'STANDING' in position:
        position = splitted_ids[-3].upper()
    elif 'LAT' in position:
        position = 'LAT'
    elif 'PA' in position:
        position = 'PA'
    elif 'AP' in position:
        position = 'AP'
    elif 'SUPINE' in position:
        return None
    elif 'CHEST' in position:
        return None
    elif 'P' == position and  splitted_ids[-3].upper() == 'A':
        position = 'AP'
    elif 'PORTRAIT' in position:
        return None
    elif 'SID' in position:
        return None
    else:
        raise_with_traceback(ValueError('Unknown position: '+position + ', for file: ' +  lineItems[0]))
    thisimage['position'] = position
    return thisimage
    
def read_filenames(pathFileTrain):
    listImage = []
    fileDescriptor = open(pathFileTrain, "r")
    
    line = True
    while line:             
      line = fileDescriptor.readline()

      #--- if not empty
      if line:
          thisimage = read_filename(line)
          if thisimage is not None:
              listImage.append(thisimage)
    fileDescriptor.close()
    return listImage

def get_name_h5_file():
    return get_name_pickle_file()[:-3]+'h5'
    #return 'all_images_224_224_prot2.2.h5'
  
def get_name_pickle_file():
    if configs['trainable_densenet']:
        if configs['use_random_crops']: 
            size_all_images = '256_256'
        else:
            size_all_images = '224_224'
    else:
        size_all_images = '7_7'
    return '/home/sci/ricbl/Documents/projects/radiology-project/pft/images_'+''.join(configs['data_to_use'])+'_' +  size_all_images + '_prot3.pkl'
  
class DatasetGenerator(Dataset):
    def __init__ (self, pathDatasetFile, transform = None):
        super(DatasetGenerator, self).__init__()
        self.listImage = pathDatasetFile
        if configs['trainable_densenet'] and configs['load_image_features_from_file']:
            self.file = h5py.File(get_name_h5_file(), 'r')
            
            #much slower
            #store = Store('all_images_224_224_prot2.2.h5df', mode="r")
            #self.df1 = store["/frames/1"]
        self.n_images = len(self.listImage[0])
        self.transform = transform
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        imageData = []
        filepath=[]
        for i in range(len(self.listImage)):
            if (configs['trainable_densenet'] and (not configs['load_image_features_from_file'])):
                imagePath = self.listImage[i]['filepath'].iloc[index]
                imageData.append(Image.open(imagePath))
                
            else:
                if configs['trainable_densenet']:
                    #a = self.listImage[i]
                    #b = a['preprocessed']
                    #print(index)
                    #print(len(b))
                    #c = b.iloc[index]
                    old_index = self.listImage[i]['preprocessed'].iloc[index]
                    
                    imageData.append(self.file['dataset_1'][old_index,...].astype('float32'))
                    
                    #much slower:
                    #examples_to_read = ['ind' + str(old_index)]
                    #imageData = self.df1.rows(examples_to_read).values.reshape(3,224,224)
                else:
                    imageData.append(self.listImage[i]['preprocessed'].iloc[index][0])
                    
            if self.transform is not None: 
                imageData[i] = self.transform(imageData[i])
                
                '''
                if i == 0:
                    imageData[i] = image_preprocessing.CropOneSideNumpy(None)(imageData[i])
                else:
                    imageData[i] = image_preprocessing.CropSideCenterNumpy(None)(imageData[i])
                '''
                
            imageData[i] = np.expand_dims(imageData[i], 0)
            filepath.append(self.listImage[i]['filepath'].iloc[index])
        imageData = np.concatenate(imageData,0)
        # for checking integrity of images and rest of data
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
        imageLabel= torch.FloatTensor(self.listImage[0][labels_columns].iloc[index])
        imageAllColumns= torch.FloatTensor(self.listImage[0][configs['all_output_columns']].iloc[index])
        
        if configs['use_extra_inputs']:
            col_extra_inputs = [col for col in self.listImage[0] if col.startswith('tobacco_status_')]+[col for col in self.listImage[0] if col.startswith('smoking_tobacco_status_')]+['binary_gender']+['age']
            extra_inputs = torch.FloatTensor(self.listImage[0][col_extra_inputs].iloc[index])
        else:
            extra_inputs = []
        
        
        #self.transform = transforms.ToTensor()
        return imageData, imageLabel, imageAllColumns, extra_inputs, filepath
    
    def __len__(self):
        return self.n_images

def get_labels():
    #selects sets of data to use
    if configs['use_set_29']:
        all_labels = pd.read_csv('./labels.csv')
        all_labels['dataset'] = '2017'
    else:
        file_root = '/usr/sci/projects/DeepLearning/Tolga_Lab/Projects/Project_JoyceSchroeder/data/data_PFT/MetaData/MetaData_'
        file_end = {'2012-2016':'2012-2016/Chest_Xray_2012-2016_TJ_clean_ResearchID_PFTValuesAndInfo_No_PHI.csv', '2017':'2017/Chest_Xray_20180316_Clem_clean_ResearchID_PFTValuesAndInfo_noPHI.csv'}
        all_labels = pd.concat([pd.read_csv(file_root + file_end[dataset]).assign(dataset=dataset) for dataset in configs['data_to_use']])
    all_labels['fev1_diff'] = all_labels['Predicted FEV1'] - all_labels['Pre-Drug FEV1']
    all_labels['fvc_diff'] = all_labels['Predicted FVC'] - all_labels['Pre-Drug FVC']

    
    

    if not configs['use_set_29']:
        all_labels.rename(index=str, columns=configs['columns_translations'], inplace = True)
    
    all_labels.fillna(value={'tobacco_status':'Not Asked', 'smoking_tobacco_status':'Never Assessed'}, inplace = True)
    all_labels.dropna(subset=['fvc_predrug'], inplace=True)
    
    all_labels['binary_gender'] = (all_labels['gender']=='Male')*1
    all_labels.replace({'tobacco_status':{
    'Quit':0,
    'Never':1,
    'Yes':2,
    'Not Asked':3,
    'Passive':4,
    'Former Smoker':5},
    'smoking_tobacco_status':
    {'Former Smoker':0,
    'Never Smoker':1,
    'Current Every Day Smoker':2,
    'Passive Smoke Exposure - Never Smoker':3,
    'Light Tobacco Smoker':4,
    'Never Assessed':5,
    'Current Some Day Smoker':6,
    'Heavy Tobacco Smoker':7,
    'Unknown If Ever Smoked':5,
    'Smoker, Current Status Unknown':5}
    }, inplace = True)
  
    all_labels = pd.concat([all_labels,pd.get_dummies(all_labels['tobacco_status'], prefix='tobacco_status').astype('float')],axis=1)
    all_labels = pd.concat([all_labels,pd.get_dummies(all_labels['smoking_tobacco_status'], prefix='smoking_tobacco_status').astype('float')],axis=1)

    all_labels[percentage_labels] = all_labels[percentage_labels] /100.
    
    if configs['use_copd_definition_as_label']:
        all_labels['copd'] = (all_labels['fev1fvc_predrug']< 0.7)*1
    
    all_labels['gold'] = metrics.get_gold(all_labels['fev1_ratio'], all_labels['fev1fvc_predrug'])
    
    if not configs['create_csv_from_dataset']:
        configs['pre_transform_labels'].set_pre_transformation_labels(all_labels)
        all_labels = configs['pre_transform_labels'].apply_transform(all_labels)
    
    return all_labels

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
    
def get_all_images():
        #selects sets of data to use
    if configs['use_set_29']:
        files_with_image_filenames = ['train.txt']
    else:
        #files_with_image_filenames = ['train2.txt']
        files_with_image_filenames = [ 'images' + str(dataset) + '.txt' for dataset in configs['data_to_use']]
    num_ftrs = None
    list_pretransforms =[]
    if (not configs['load_image_features_from_file']) or configs['trainable_densenet']:
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
        
        if configs['use_random_crops']: 
            resize_size = 256
        else:
            resize_size = 224
        list_pretransforms = list_pretransforms + [     image_preprocessing.Convert16BitToFloat(),
                  image_preprocessing.CropBiggestCenteredInscribedSquare(),
                  transforms.Resize(size=(resize_size)), #transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  normalize]
        
        all_images = []
        for file_with_image_filenames in files_with_image_filenames:
            all_images = all_images + read_filenames('/home/sci/ricbl/Documents/projects/radiology-project/pft/' + file_with_image_filenames)
        
        all_images = pd.DataFrame(all_images)
        
        '''
        transformSequence = transforms.Compose(list_pretransforms)
        all_images = image_preprocessing.preprocess_images(all_images, transformSequence)
        #h5f = h5py.File(get_name_h5_file(), 'w')
        #h5f.create_dataset('dataset_1', data=np.array(all_images.preprocessed.tolist()))
        #h5f.close()
        
        
        with open(get_name_pickle_file(), 'wb') as f:
            pickle.dump(all_images, f, protocol=2)
        '''
        
        
    if (not configs['trainable_densenet']) and (not configs['load_image_features_from_file']):
        chexnetModel = model_loading.load_pretrained_chexnet()
        num_ftrs = chexnetModel.module.model.classifier[0].in_features
        
        chexnetModel.module.model.classifier = torch.nn.Sequential()
        chexnetModel = chexnetModel.cuda()  

        list_pretransforms.append(image_preprocessing.ChexnetEncode(chexnetModel))
            
        transformSequence = transforms.Compose(list_pretransforms)
        
        all_images = image_preprocessing.preprocess_images(all_images, transformSequence)

        with open(get_name_pickle_file(), 'wb') as f:
            pickle.dump(all_images, f, protocol=2)
    elif (not configs['trainable_densenet']) and (configs['load_image_features_from_file']):
        all_images = pd.read_pickle(get_name_pickle_file())
        image_shape = all_images['preprocessed'][0][0].shape
        num_ftrs = image_shape[0]
    all_images['image_index'] = all_images.index
    
    if (configs['trainable_densenet'] and (configs['load_image_features_from_file'])):
        all_images['preprocessed'] = all_images.index
    return all_images, num_ftrs, list_pretransforms
        
def get_images():

    all_images, num_ftrs, list_pretransforms = get_all_images()
    
    sets_of_images = []
    if not configs['create_csv_from_dataset']:
        sets_of_images.append(all_images[all_images['position'].isin(configs['positions_to_use'])])
    
    if configs['use_lateral']:
        if configs['create_csv_from_dataset']:
            sets_of_images.append(all_images[all_images['position'].isin(['PA'])])
            sets_of_images.append(all_images[all_images['position'].isin(['AP'])])
        sets_of_images.append(all_images[all_images['position'].isin(['LAT'])])
    list_transforms = []
    if configs['trainable_densenet'] and (not configs['load_image_features_from_file']):
        list_transforms = list_transforms + list_pretransforms + [image_preprocessing.ToNumpy()]
    train_list_transforms = list_transforms
    test_list_transforms = list_transforms
    if configs['trainable_densenet']:
        train_list_transforms = train_list_transforms + [image_preprocessing.RandomHorizontalFlipNumpy()]
    else: 
        train_list_transforms = train_list_transforms + [image_preprocessing.RandomHorizontalFlipNumpy()]
    if configs['use_random_crops'] and configs['trainable_densenet']:
        train_list_transforms = train_list_transforms+ [image_preprocessing.RandomCropNumpy(224)]
        test_list_transforms = test_list_transforms+ [image_preprocessing.CenterCropNumpy(224)]
    trainTransformSequence = transforms.Compose(train_list_transforms)
    testTransformSequence = transforms.Compose(test_list_transforms)
    return sets_of_images, trainTransformSequence, testTransformSequence, num_ftrs
