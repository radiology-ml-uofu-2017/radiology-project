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

def get_name_pickle_file():
    if configs['trainable_densenet']:
        size_all_images = '224_224'
    elif configs['remove_pre_avg_pool']:
        size_all_images = '7_7'
    else:
        size_all_images = '1_1'
    return '/home/sci/ricbl/Documents/projects/radiology-project/pft/all_images_' +  size_all_images + '_prot2.pkl'
  
class DatasetGenerator(Dataset):
    def __init__ (self, pathDatasetFile, transform = None):
        super(DatasetGenerator, self).__init__()
        self.listImage = pathDatasetFile
        
        if configs['trainable_densenet'] and configs['load_image_features_from_file']:
            self.file = h5py.File('./all_images_224_224_prot2.2.h5', 'r')
            
            #much slower
            #store = Store('all_images_224_224_prot2.2.h5df', mode="r")
            #self.df1 = store["/frames/1"]
        self.n_images = len(self.listImage[0])
        self.transform = transform
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        imageData = []
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
            imageData[i] = np.expand_dims(imageData[i], 0)
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
        imageAllColumns= torch.FloatTensor(self.listImage[0][configs['all_input_columns']].iloc[index])
        #self.transform = transforms.ToTensor()
        
        return imageData, imageLabel, imageAllColumns
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        return self.n_images

def get_labels():
    #selects sets of data to use
    if configs['use_set_29']:
        file_with_labels = './labels.csv'
    else:
        file_with_labels = '/usr/sci/projects/DeepLearning/Tolga_Lab/Projects/Project_JoyceSchroeder/data/data_PFT/data_EDW_clean_Clem/Chest_Xray_20180316_Clem_clean_ResearchID_PFTValuesAndInfo_noPHI.csv'
    
    all_labels = pd.read_csv(file_with_labels)
    if not configs['use_set_29']:
        all_labels.rename(index=str, columns=configs['columns_translations'], inplace = True)
    
    all_labels.dropna(subset=['fvc_predrug'], inplace=True)

    all_labels[percentage_labels] = all_labels[percentage_labels] /100.
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
        file_with_image_filenames = 'train.txt'
    else:
        file_with_image_filenames = 'train2.txt'
    num_ftrs = 1024
    
    if (not configs['load_image_features_from_file']) or configs['trainable_densenet']:
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
        
        list_pretransforms =[     image_preprocessing.Convert16BitToFloat(),
                  image_preprocessing.CropBiggestCenteredInscribedSquare(),
                  transforms.Resize(size=(224)), 
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  normalize]
        
        all_images = read_filenames('/home/sci/ricbl/Documents/projects/radiology-project/pft/' + file_with_image_filenames)
       
    if (not configs['trainable_densenet']) and (not configs['load_image_features_from_file']):
        chexnetModel = models.load_pretrained_chexnet()
        num_ftrs = chexnetModel.module.densenet121.classifier[0].in_features
        
        chexnetModel.module.densenet121.classifier = nn.Sequential()
        chexnetModel = chexnetModel.cuda()  
        list_pretransforms.append(image_preprocessing.ChexnetEncode(chexnetModel))
            
        transformSequence = transforms.Compose(list_pretransforms)
        
        all_images = image_preprocessing.preprocess_images(all_images, transformSequence)
        del transformSequence
        del chexnetModel
        with open(get_name_pickle_file(), 'wb') as f:
            pickle.dump(all_images, f, protocol=2)
    elif (not configs['trainable_densenet']) and (configs['load_image_features_from_file']):
        all_images = pd.read_pickle(get_name_pickle_file())

    all_images['image_index'] = all_images.index
    
    if (configs['trainable_densenet'] and (configs['load_image_features_from_file'])):
        all_images['preprocessed'] = all_images.index
    return all_images, num_ftrs
        
def get_images():

    all_images, num_ftrs = get_all_images()
    
    sets_of_images = []
    sets_of_images.append(all_images[all_images['position'].isin(configs['positions_to_use'])])
    
    if configs['use_lateral']:
        sets_of_images.append(all_images[all_images['position'].isin(['LAT'])])
    
    if configs['trainable_densenet'] and (not configs['load_image_features_from_file']):
        transformSequence = transforms.Compose(list_pretransforms)
    elif configs['trainable_densenet']:
        transformSequence = transforms.Compose([image_preprocessing.RandomHorizontalFlipNumpy()])
    else: 
        transformSequence = transforms.Compose([image_preprocessing.RandomHorizontalFlipNumpy()])
        
    return sets_of_images, transformSequence, num_ftrs
