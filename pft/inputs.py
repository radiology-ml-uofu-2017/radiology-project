import os
from future.utils import raise_with_traceback
import pandas as pd
from configs import configs
from torch.utils.data import Dataset
import torch
#from h5df import Store
import h5py
import torchvision.transforms as transforms
import image_preprocessing
import numpy as np
import metrics
from PIL import Image
import model_loading
import torchvision
import copy
import glob
import math
from PIL import ImageMath
from collections import OrderedDict
try:
    import cPickle as pickle
except:
    import _pickle as pickle

percentage_labels = configs['percentage_labels']

labels_columns = configs['get_labels_columns']

def write_filenames():
    #pathFileTrain = '/home/sci/ricbl/Documents/projects/radiology-project/pft/train2.txt'
    #pathPngs = '/usr/sci/projects/DeepLearning/Tolga_Lab/Projects/Project_JoyceSchroeder/data/data_PFT/'
    #pathFileTrain = '/home/sci/ricbl/Documents/projects/temp_radiology/radiology-project/pft/train_spiro.txt'
    #pathPngs = '/usr/sci/projects/DeepLearning/Tolga_Lab/Projects/Project_JoyceSchroeder/data/data_SPIROMICS/'
    #command = 'find ' + pathPngs + ' -type f -name "*.png" > ' + pathFileTrain
    pathFileTrain = '/home/sci/ricbl/Documents/projects/temp_radiology/radiology-project/pft/train_chexpert.txt'
    pathPngs = configs['chestxpert_path']
    command = 'find ' + pathPngs + ' -type f -name "*.jpg" > ' + pathFileTrain
    
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

def read_filename_chexpert(line):
    thisimage = {}

    lineItems = line.split()
    thisimage['filepath'] = lineItems[0]
    splitted_filepath = thisimage['filepath'].replace('\\','/').split("/")
    splitted_ids = splitted_filepath[-1].replace('-','_').replace('.','_').split('_')
    thisimage['subjectid'] = int(splitted_filepath[-3][7:])
    thisimage['crstudy'] = int(splitted_filepath[-2][5:])
    thisimage['scanid'] = int(splitted_ids[-3][4:])
    position = splitted_ids[-2].upper()
    if 'LATERAL' in position:
        position = 'LAT'
    elif 'FRONTAL' in position:
        position = 'PA'
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

def get_name_h5_file(train = True):
    return get_name_pickle_file(train)[:-3]+'h5'
    #return 'all_images_224_224_prot2.2.h5'

def get_name_pickle_file(split_string):
    if configs['trainable_densenet']:
        size_all_images = '256_256'
        # if configs['use_random_crops']:
        #     size_all_images = '256_256'
        # else:
        #     size_all_images = '224_224'
    else:
        size_all_images = '7_7'
    if configs['machine_to_use'] == 'titan':
        folder = '/scratch_ssd/ricbl/pft'
    elif configs['machine_to_use'] == 'atlas':
        folder = '/scratch/ricbl/pft'
    else:
        folder = configs['local_directory']
    if configs['use_cut_restrictive']:
        return folder + '/images_restrictive.' + split_string + '.2.pkl'
    else:
        return folder + '/images_' +''.join(configs['data_to_use'])+'_' +  size_all_images + '_prot3.pkl'

class DatasetGenerator(Dataset):
    def __init__ (self, pathDatasetFile, transform = None, train = False, segmentation_features_file = None, split = 'train'):
        super(DatasetGenerator, self).__init__()
        self.listImage = pathDatasetFile
        self.file = None
        self.segmentation = None
        self.n_images = len(self.listImage[0])
        self.transform = transform
        self.train = train
        self.split = 'val' if split=='validation' else ('train' if split == 'eval_train' else split)
        assert(not configs['use_lateral'] or not configs['use_cut_restrictive'])
        self.load_images_from_file = configs['trainable_densenet'] and configs['load_image_features_from_file']
        self.load_segmentation = (configs['use_cut_restrictive'] and not configs['load_image_features_from_file']) or ((configs['use_unet_segmentation'] or configs['register_with_segmentation'] or  configs['calculate_segmentation_features']) and configs['segmentation_in_loading'])
        if self.load_images_from_file and self.file is None and configs['load_dataset_to_memory']:
            self.load_h5py_file()
            self.loaded_images = []
            for i in range(model_loading.get_qt_inputs()[0]):
                self.loaded_images.append([self.read_one_example_from_h5_file(index, i) for index in range(self.n_images)])
        
    #--------------------------------------------------------------------------------
    # def __del__(self):  
    #     if self.file is not None:
    #         self.file.close()
    #         self.file = None
    #         print('File closed')
    #     if self.segmentation is not None:
    #         self.segmentation.close()
    #         self.segmentation = None
    # 
    # def __delete__(self):  
    #     if self.file is not None:
    #         self.file.close()
    #         self.file = None
    #     if self.segmentation is not None:
    #         self.segmentation.close()
    #         self.segmentation = None
        
    def load_h5py_file(self):
        if configs['machine_to_use'] == 'titan':
            folder = '/scratch_ssd/ricbl/pft'
        elif configs['machine_to_use'] == 'atlas':
            folder = '/scratch/ricbl/pft'
        else:
            folder = configs['local_directory']
        assert(not configs['magnification_input']>1 or not configs['use_cut_restrictive'])
        if not configs['magnification_input']>1:
            self.file = h5py.File(get_name_h5_file(self.split), 'r', swmr = True)
        else:
            assert(configs['magnification_input']<=4)
            self.file = h5py.File(folder + '/images_2012-20162017_1024_1024.prot3.h5', 'r', swmr = True)
    
    def read_one_example_from_h5_file(self, index, i):
        if configs['use_cut_restrictive']:
            if not configs['two_inputs'] :
                dataset_name = configs['image_to_use_for_one_input_restrictive']
            elif i==0:
                dataset_name = 'frontal'
            else:
                dataset_name = 'zoom'
            return configs['unary_input_multiplier']*self.file[dataset_name][index,...].astype('float32')
        else:
            return configs['unary_input_multiplier']*self.file['dataset_1'][self.listImage[i]['preprocessed'].iloc[index],...].astype('float32')
        
    def __getitem__(self, index):
        if self.load_images_from_file and self.file is None and not configs['load_dataset_to_memory']:
            self.load_h5py_file()
        
        if self.load_segmentation and self.segmentation is None:
            if configs['use_precalculated_segmentation_features']:
                self.segmentation = h5py.File(segmentation_features_file, 'r', swmr = True)
            else:
                self.segmentation = model_loading.SegmentationNetwork().cuda()
        if not isinstance(index, int):
            index = int(index)
        imageData = []
        filepath=[]
        vectorized_features = 0
        for i in range(model_loading.get_qt_inputs()[0]):
            if (configs['use_cut_restrictive'] and i == 1 and (not configs['load_image_features_from_file'])):
                 imageData.append(cut_corner)
            else:
                if (configs['trainable_densenet'] and (not configs['load_image_features_from_file'])):
                    imagePath = self.listImage[i]['filepath'].iloc[index]
                    # if i == 0:
                    #     imagePath = '/usr/sci/projects/DeepLearning/Tolga_Lab/Projects/Project_JoyceSchroeder/data/data_PFT/CR_Orthanc_PNG_2017/PFT_001620/PFT_001620_CRStudy_01/PFT_001620_CRStudy_01_ScanID_1-W_Chest_PA.png'
                    # else:
                    #     imagePath = '/usr/sci/projects/DeepLearning/Tolga_Lab/Projects/Project_JoyceSchroeder/data/data_PFT/CR_Orthanc_PNG_2017/PFT_001620/PFT_001620_CRStudy_01/PFT_001620_CRStudy_01_ScanID_2-W_Chest_Lat.png'
                    # if i == 0:
                    #     imagePath = '/usr/sci/projects/DeepLearning/Tolga_Lab/Projects/Project_JoyceSchroeder/data/data_PFT/CR_Orthanc_PNG_2017/PFT_001126/PFT_001126_CRStudy_02/PFT_001126_CRStudy_02_ScanID_1-W_CHEST_PA.png'
                    # else:
                    #     imagePath = '/usr/sci/projects/DeepLearning/Tolga_Lab/Projects/Project_JoyceSchroeder/data/data_PFT/CR_Orthanc_PNG_2017/PFT_001126/PFT_001126_CRStudy_02/PFT_001126_CRStudy_02_ScanID_2-W_CHEST_LAT.png'
                    imageData.append(Image.open(imagePath))

                else:
                    if configs['trainable_densenet']:
                        #a = self.listImage[i]
                        #b = a['preprocessed']
                        #print(index)
                        #print(len(b))
                        #c = b.iloc[index]
                        
                        if not configs['load_dataset_to_memory']:
                            imageData.append(self.read_one_example_from_h5_file(index, i))
                        else:
                            imageData.append(self.loaded_images[i][index])
                        #much slower:
                        #examples_to_read = ['ind' + str(old_index)]
                        #imageData = self.df1.rows(examples_to_read).values.reshape(3,224,224)
                    else:
                        imageData.append(self.listImage[i]['preprocessed'].iloc[index][0])
            original_image = imageData[i]
            if self.transform[i] is not None:
                imageData[i] = self.transform[i](imageData[i])
                #torchvision.utils.save_image(imageData[i], 'test_corner_t'+str(i)+'_'+configs['timestamp']+'.png')
                if (configs['use_cut_restrictive'] and i ==0 and (not configs['load_image_features_from_file'])):
                    transforms2 = copy.deepcopy(self.transform[i])
                    #transforms2.transforms = transforms2.transforms[:2] + transforms2.transforms[3:6] + transforms2.transforms[-1:]
                    transforms2.transforms = transforms2.transforms[:2] + transforms2.transforms[3:4] + transforms2.transforms[5:6] + transforms2.transforms[-1:]
                    original_image = transforms2(original_image)
                    #torchvision.utils.save_image(imageData[i] , './test_transform_cut' + str(i)+'.png')
                '''
                if i == 0:
                    imageData[i] = image_preprocessing.CropOneSideNumpy(None)(imageData[i])
                else:
                    imageData[i] = image_preprocessing.CropSideCenterNumpy(None)(imageData[i])
                '''

            #imageData[i] = np.expand_dims(imageData[i], 0)
            if configs['use_cut_restrictive']:
                listImage = self.listImage[0]
            else:
                listImage = self.listImage[i]
            filepath.append(listImage['filepath'].iloc[index])

            if (configs['use_cut_restrictive'] and i ==0 and (not configs['load_image_features_from_file'])) or ((configs['use_unet_segmentation'] or configs['register_with_segmentation'] or configs['calculate_segmentation_features']) and configs['segmentation_in_loading'] and not configs['use_precalculated_segmentation_features']):
                #assert(not configs['register_with_segmentation']) #TODO: implement registration somewhere
                if i == 0 or configs['use_unet_segmentation_for_lateral']:
                    imageData[i], features = self.segmentation(imageData[i].cuda().unsqueeze(0))
                    if configs['use_cut_restrictive'] and i ==0:
                        if ('right' not in features[0].keys()) or (features[0]['right'] is None):
                            print('oi4')
                            print(index)
                            right_lung_corner = (185, 25)
                        else:
                            right_lung_corner = (features[0]['right']['row_corner_0'], features[0]['right']['column_corner_0'])
                        margin = 8
                        margin_1 = (223-right_lung_corner[0]) if ((right_lung_corner[0]+margin)*original_image.size(1)/224>=original_image.size(1)) else margin
                        margin = 4
                        margin_2 = (right_lung_corner[1]) if ((right_lung_corner[1]-margin)<0) else margin
                        # print(features[0]['right'].keys())
                        # print(224*3/(features[0]['right']['area_lung']*((original_image.size(1)/224)**2)))
                        # print(224*3/(features[0]['right']['distance']*(original_image.size(1)/224)))
                        if ('right' not in features[0].keys()) or (features[0]['right'] is None):
                            square_size = 680
                        else:
                            square_size = max(
                                            int(min(features[0]['right']['distance'], 112)*math.cos(features[0]['right']['angle'])*0.9*(original_image.size(1)/224)),
                                            int(min(features[0]['right']['distance'], 112)*math.sin(features[0]['right']['angle'])*1.5*(original_image.size(1)/224))
                                            )#224*3)
                        #cut_corner = Image.fromarray((original_image[:,int(right_lung_corner[0]*original_image.size(1)/224)-square_size+int(margin_1*original_image.size(1)/224):int(right_lung_corner[0]*original_image.size(1)/224)+int(margin_1*original_image.size(1)/224), int(right_lung_corner[1]*original_image.size(2)/224)-int(margin_2*original_image.size(2)/224):int(right_lung_corner[1]*original_image.size(2)/224)+square_size-int(margin_2*original_image.size(2)/224)].cpu().permute([1,2,0]).numpy()*65535).astype(dtype=np.uint16), 'I;16')
                        cut_corner = Image.fromarray((original_image[:,int(right_lung_corner[0]*original_image.size(1)/224)-square_size+int(margin_1*original_image.size(1)/224):int(right_lung_corner[0]*original_image.size(1)/224)+int(margin_1*original_image.size(1)/224), int(right_lung_corner[1]*original_image.size(2)/224)-int(margin_2*original_image.size(2)/224):int(right_lung_corner[1]*original_image.size(2)/224)+square_size-int(margin_2*original_image.size(2)/224)].cpu()[0,:,:].numpy()*255*255).astype(dtype=np.uint16), 'I;16').convert('I')
                        # cut_corner.save('test_corner_pil.png', 'png')
                        # print(cut_corner)
                        # print(np.max(cut_corner))
                        # print(np.min(cut_corner))
                        #imageData[1] = cut_corner
                        # torchvision.utils.save_image(torch.tensor(cut_corner), 'test_corner_'+configs['timestamp']+'.png')
                        # torchvision.utils.save_image(torch.tensor(original_image), 'test_corner_original_'+configs['timestamp']+'.png')
                        # 1/0
                    features = features[0]
                    
                    if configs['register_with_segmentation']:
                        grid = torch.nn.functional.affine_grid(features['theta'].unsqueeze(0), imageData[i].size())
                        imageData[i] = torch.nn.functional.grid_sample(imageData[i], grid)
                        
                    imageData[i] = imageData[i].squeeze(0).cpu()
                    
                    vectorized_features = OrderedDict()
                    vectorized_features.update({'theta_0_0': features['theta'][0][0].cpu(), 'theta_1_0': features['theta'][1][0].cpu(), \
                    'theta_0_1': features['theta'][0][1].cpu(),'theta_1_1': features['theta'][1][1].cpu(), \
                    'theta_0_2': features['theta'][0][2].cpu(),'theta_1_2': features['theta'][1][2].cpu()})
                    feature_columns = ['row_corner_0','column_corner_0','row_corner_1','column_corner_1','angle','distance','area_lung','orientation_lung','lung_elongation','bounding_box_occupation','lung_eccentricity','average_intensity_lung','average_upper_half_intensity_lung','curl_diaphragm','ellipse_average_intensity','watershed_diaphragm_score','bounding_box_upper_corner_void_occupation','upper_corner_void_convexity']
                    if 'right' in features.keys() and features['right'] is not None :
                        vectorized_features.update({feature_column+'-right':features['right'][feature_column] for feature_column in feature_columns})
                    else:
                        vectorized_features.update({feature_column+'-right':0 for feature_column in feature_columns})
                    if 'left' in features.keys() and features['left'] is not None :
                        vectorized_features.update({feature_column+'-left':features['left'][feature_column] for feature_column in feature_columns})
                    else:
                        vectorized_features.update({feature_column+'-left':0 for feature_column in feature_columns})
                
        if (configs['use_unet_segmentation'] or configs['register_with_segmentation'] or configs['calculate_segmentation_features']) and configs['segmentation_in_loading']:
            if not configs['use_precalculated_segmentation_features']:
                feature_columns_tmp = ['row_corner_0','column_corner_0','row_corner_1','column_corner_1','angle','distance','area_lung','orientation_lung','lung_elongation','bounding_box_occupation','lung_eccentricity','average_intensity_lung','average_upper_half_intensity_lung','curl_diaphragm','ellipse_average_intensity','watershed_diaphragm_score','bounding_box_upper_corner_void_occupation','upper_corner_void_convexity']
                features_columns = ['theta_0_0','theta_1_0','theta_0_1','theta_1_1','theta_0_2','theta_1_2'] + [feature_column + '-right' for feature_column in feature_columns_tmp]+ [feature_column + '-left' for feature_column in feature_columns_tmp]
                vectorized_features = [float(vectorized_features[key]) for key in features_columns]
            else:
                vectorized_features = self.segmentation['dataset_1'][index,...]
        vectorized_features = torch.tensor(vectorized_features)

        #imageData = np.concatenate(imageData,0)
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
        if configs['subtract_0.7']:
            imageLabel = (imageLabel - 0.7)/0.1
        if (configs['relative_noise_to_add_to_label'] is not None) and (self.train):
            imageLabel = torch.FloatTensor(np.random.normal(imageLabel, imageLabel * configs['relative_noise_to_add_to_label']))
        imageAllColumns= torch.FloatTensor(self.listImage[0][configs['all_output_columns']].iloc[index])
        if configs['use_extra_inputs']:
            if configs['use_smokeless_history']:
                col_extra_inputs = [col for col in self.listImage[0] if col.startswith('tobacco_status_')]+[col for col in self.listImage[0] if col.startswith('smoking_tobacco_status_')]+['binary_gender']+['age']
            else:
                col_extra_inputs = [col for col in self.listImage[0] if col.startswith('smoking_tobacco_status_')]+['binary_gender']+['age']
            extra_inputs = torch.FloatTensor(self.listImage[0][col_extra_inputs].iloc[index])
            extra_inputs[-1] = extra_inputs[-1]/100.
        else:
            extra_inputs = []
        #self.transform = transforms.ToTensor()
        # torchvision.utils.save_image((imageData[0]+2.5)/5, 'testourpftft'+str(index)+'.png')
        # torchvision.utils.save_image((imageData[1]+2.5)/5, 'testourpftlt'+str(index)+'.png')
        # 1/0
        #print('our pft stats:')
        #print(torch.max(imageData[0]))
        #print(torch.min(imageData[0]))
        #print(torch.mean(imageData[0]))
        # if self.file is not None:
        #     self.file.close()
        #     self.file = None
        #     print('File closed')
        # if self.segmentation is not None:
        #     self.segmentation.close()
        #     self.segmentation = None

        # print(torch.max(1-image_preprocessing.BatchUnNormalizeTensor(configs['normalization_mean'],configs['normalization_std'])(imageData[0])))
        # print(torch.min(1-image_preprocessing.BatchUnNormalizeTensor(configs['normalization_mean'],configs['normalization_std'])(imageData[0])))
        # print(self.split)
        # print(self.listImage[0]['filepath'].iloc[index])
        # print(imageLabel)
        # torchvision.utils.save_image(image_preprocessing.BatchUnNormalizeTensor(configs['normalization_mean'],configs['normalization_std'])(imageData[0]), '224frontal'+str(index)+'.png')
        # if configs['two_inputs']:
        #     torchvision.utils.save_image(image_preprocessing.BatchUnNormalizeTensor(configs['normalization_mean'],configs['normalization_std'])(imageData[1]), '224lateral'+str(index)+'.png')
        # 1/0
        return imageData[0], ([] if model_loading.get_qt_inputs()[0]==1 else imageData[1]), imageLabel, imageAllColumns, extra_inputs, filepath, vectorized_features, index, self.listImage[0][configs['example_identifier_columns']].iloc[index].values

    def __len__(self):
        return self.n_images

def get_transformations(original_image, transform):
    images = []
    if configs['use_transformation_loss']:
        n_transformations = configs['transformation_group_size']
    else:
        n_transformations = 1
    for i in range(n_transformations):
        if transform is not None:
            image = transform(original_image)
        else:
            image = original_image
        images.append(-image.unsqueeze(0))
    images = torch.cat(images, dim = 0)
    if not configs['use_transformation_loss']:
        images = images.squeeze(0)
    return images

class Chestxray14Dataset(Dataset):
    def __init__(self, transform = None):
        super(Chestxray14Dataset, self).__init__()
        assert(configs['use_chexpert'] or not configs['use_lateral'])
        self.files = []
        if configs['use_chexpert']:
            all_images = []
            
            fileDescriptor = open(configs['local_directory'] + '/' + 'train_chexpert.txt', "r")

            line = True
            while line:
              line = fileDescriptor.readline()

              #--- if not empty
              if line:
                  thisimage = read_filename_chexpert(line)
                  if thisimage is not None:
                      self.files.append(thisimage)
            fileDescriptor.close()
            
            # for filename in glob.glob(configs['chestxpert_path']+'/patient*/study*/*.jpg'):
            #     thisimage = read_filename_chexpert(filename)
            #     if thisimage is not None:
            #         self.files.append(thisimage)
        else:
            for filename in glob.glob(configs['chestxray14_path']+'/*.png'):
                self.files.append({'filepath':filename})
        if configs['use_random_crops']:
            resize_size = 256
        else:
            resize_size = 224
        normalize = transforms.Normalize(configs['normalization_mean'],
                                    configs['normalization_std'])
        list_pretransforms = [transforms.Resize(size=(resize_size)), transforms.ToTensor(), image_preprocessing.ToNumpy()]
        if configs['use_chexpert']:
                list_pretransforms = [image_preprocessing.CropBiggestCenteredInscribedSquare()] + list_pretransforms
        self.transform = transforms.Compose(list_pretransforms + transform[0].transforms + [normalize])
        self.files = pd.DataFrame(self.files)
        if configs['use_chexpert']:
            self.cases_to_use = pd.merge(self.files[self.files['position']=='LAT'], self.files[self.files['position']=='PA'], on=['subjectid', 'crstudy'], how = 'inner')  
        else:
            self.cases_to_use = self.files
    def __getitem__(self, index):
        
        if not configs['use_chexpert']:
            filepath_frontal = self.files.iloc[index,0]
        else:
            subjectid = self.cases_to_use['subjectid'].iloc[index]
            crstudy = self.cases_to_use['crstudy'].iloc[index]
            selected_cases = self.files[(self.files['subjectid'] == subjectid) & (self.files['crstudy'] == crstudy)]
            filepath_lateral = selected_cases[self.files['position']=='LAT']['filepath'].iloc[0]
            filepath_frontal = selected_cases[self.files['position']=='PA']['filepath'].iloc[0]
        original_frontal_image = Image.open(filepath_frontal).convert('RGB') #TODO check if they are inversed; follow same steps to inverse as the rest; normalize the inversed and then negative
        frontal_images = get_transformations(original_frontal_image, self.transform)
        if configs['use_lateral']:
            original_lateral_image = Image.open(filepath_lateral).convert('RGB') #TODO check if they are inversed; follow same steps to inverse as the rest; normalize the inversed and then negative
            lateral_images = get_transformations(original_lateral_image, self.transform) #TODO: lateral getting flipped
        else:
            lateral_images = []

        # torchvision.utils.save_image((frontal_images+2.5)/5, 'testchestxray14fr'+str(index)+'.png')
        # torchvision.utils.save_image((lateral_images+2.5)/5, 'testchestxray14lt'+str(index)+'.png')
        return frontal_images, lateral_images
        
    def __len__(self):
        return len(self.cases_to_use)

class _IteratorLoaderDifferentSizes:
    def __init__(self, bigLoader, smallLoader):
        self.bigLoader = bigLoader
        self.smallLoader = smallLoader
        if smallLoader.batch_size*len(smallLoader)> len(bigLoader)*bigLoader.batch_size:
            raise_with_traceback(ValueError('bigLoader (first constructor argument) should be bigger than smallLoader (second argument).'))
        if len(smallLoader)> len(bigLoader):
            raise_with_traceback(ValueError('With these batch sizes, not all supervised examples will be applied in one epoch'))        
        self.iterBigLoader = iter(self.bigLoader)
        self.iterSmallLoader = iter(self.smallLoader)

    def __iter__(self):
        return self
    
    def nextI(self, this_iter):
        return next(this_iter,None)
    
    def __next__(self):
      
        current_batch_BigLoader = self.nextI(self.iterBigLoader)
        if current_batch_BigLoader is None:
            raise StopIteration
        
        current_batch_SmallLoader= self.nextI(self.iterSmallLoader)
        if current_batch_SmallLoader is None:
            raise StopIteration
        return current_batch_BigLoader, current_batch_SmallLoader
      
    next = __next__ 

class loaderDifferentSizes:
    def __init__(self, bigLoader, smallLoader):
        self.bigLoader = bigLoader
        self.smallLoader = smallLoader
    
    def __iter__(self):
        self.iterator = _IteratorLoaderDifferentSizes(self.bigLoader, self.smallLoader)
        return self.iterator

def get_labels():
    #selects sets of data to use
    if configs['use_set_29']:
        all_labels = pd.read_csv('./labels.csv')
        all_labels['dataset'] = '2017'
    elif configs['use_set_spiromics']:
        all_labels = pd.read_csv('./SPIROMICS_PFTValuesAndInfo_No_PHI.csv')
        all_labels['dataset'] = 'SPIROMICS'
    else:
        file_root = configs['meta_file_root']
        #file_end = {'2012-2016':'2012-2016/Chest_Xray_2012-2016_TJ_clean_ResearchID_PFTValuesAndInfo_WithDate_No_PHI.csv', '2017':'2017/Chest_Xray_20180316_Clem_clean_ResearchID_PFTValuesAndInfo_WithDate_noPHI.csv'}
        #all_labels = pd.concat([pd.read_csv(file_root + file_end[dataset]).assign(dataset=dataset) for dataset in configs['data_to_use']])
        file_end = 'All/Chest_Xray_Main_TJ_clean_ResearchID_PFTValuesAndInfo_WithDate_NoPHI.csv'
        all_labels = pd.read_csv(file_root + file_end) 
        all_labels['dataset'] = all_labels.apply(lambda row: '2017' if row['PFT_DATE_YearOnly']==2017 else '2012-2016', axis=1)
        all_labels = all_labels.loc[all_labels['dataset'].isin(configs['data_to_use'])]
    all_labels['fev1_diff'] = all_labels['Predicted FEV1'] - all_labels['Pre-Drug FEV1']
    all_labels['fvc_diff'] = all_labels['Predicted FVC'] - all_labels['Pre-Drug FVC']

    if not configs['use_set_29']:
        all_labels.rename(index=str, columns=configs['columns_translations'], inplace = True)

    all_labels.fillna(value={'tobacco_status':'Not Asked', 'smoking_tobacco_status':'Never Assessed'}, inplace = True)
    all_labels.dropna(subset=['fvc_predrug'], inplace=True)

    all_labels['binary_gender'] = (all_labels['gender']=='Male')*1
    
    replacement_dicts = {}
    if configs['use_smokeless_history']:
        tobacco_status_categories = {
        'Quit':0,
        'Never':1,
        'Yes':2,
        'Not Asked':3,
        'Passive':4}
        replacement_dicts['tobacco_status'] = tobacco_status_categories
    
    smoking_tobacco_status_categories = {'Former Smoker':0,
    'Never Smoker':1,
    'Current Every Day Smoker':2,
    'Passive Smoke Exposure - Never Smoker':3,
    'Light Tobacco Smoker':4,
    'Never Assessed':5,
    'Current Some Day Smoker':6,
    'Heavy Tobacco Smoker':7,
    'Unknown If Ever Smoked':5,
    'Smoker, Current Status Unknown':5}
    replacement_dicts['smoking_tobacco_status'] = smoking_tobacco_status_categories
    
    all_labels.replace(replacement_dicts, inplace = True)
    
    if configs['use_smokeless_history']:
        all_labels = pd.concat([all_labels,pd.get_dummies(all_labels['tobacco_status'], prefix='tobacco_status').T.reindex(['tobacco_status_' + str(value) for value in range(1+max(tobacco_status_categories.values()))]).T.fillna(0).astype('float')],axis=1)
    all_labels = pd.concat([all_labels,pd.get_dummies(all_labels['smoking_tobacco_status'], prefix='smoking_tobacco_status').T.reindex(['smoking_tobacco_status_' + str(value) for value in range(1+max(smoking_tobacco_status_categories.values()))]).T.fillna(0).astype('float').astype('float')],axis=1)

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
    elif configs['use_set_spiromics']:
        files_with_image_filenames = ['train_spiro.txt']
    else:
        #files_with_image_filenames = ['train2.txt']
        files_with_image_filenames = [ 'images' + str(dataset) + '.txt' for dataset in configs['data_to_use']]
    num_ftrs = None
    list_pretransforms =[]
    if (not configs['load_image_features_from_file']) or configs['trainable_densenet']:
        normalize = transforms.Normalize(configs['normalization_mean'],
                                            configs['normalization_std'])

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
            all_images = all_images + read_filenames(configs['local_directory'] + '/' + file_with_image_filenames)
        n_images = len(all_images)
        all_images = pd.DataFrame(all_images)

        # transformSequence = transforms.Compose(list_pretransforms)
        # all_images = image_preprocessing.preprocess_images(all_images, transformSequence)

        # h5f = h5py.File('/scratch/ricbl/pft/images_restrictive.h5', 'w')
        # h5f.create_dataset('dataset_1', (n_images, 3, 224, 224))
        # all_images = image_preprocessing.preprocess_images_and_save(all_images, transformSequence, h5f)
        # h5f.close()
        # 1/0

        #h5f = h5py.File(get_name_h5_file(), 'w')
        # h5f = h5py.File('/scratch_ssd/pft/images_2012-20162017_1024_1024.prot3.h5', 'w')
        # h5f.create_dataset('dataset_1', data=np.array(all_images.preprocessed.tolist()))
        # h5f.close()
        # 1/0

        # with open(get_name_pickle_file(), 'wb') as f:
        #     pickle.dump(all_images, f, protocol=2)

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

        if configs['use_lateral'] and not (configs['use_cut_restrictive']):
            sets_of_images.append(all_images[all_images['position'].isin(['LAT'])])
    else:
        sets_of_images.append(all_images[all_images['position'].isin(['PA'])])
        sets_of_images.append(all_images[all_images['position'].isin(['AP'])])
        sets_of_images.append(all_images[all_images['position'].isin(['LAT'])])
    list_transforms = []
    if configs['trainable_densenet'] and (not configs['load_image_features_from_file']):
        list_transforms = list_transforms + list_pretransforms + [image_preprocessing.ToNumpy()]
    train_list_transforms = [copy.deepcopy(list_transforms)]
    test_list_transforms = [copy.deepcopy(list_transforms)]
    if configs['two_inputs']:
        train_list_transforms.append(copy.deepcopy(list_transforms))
        test_list_transforms.append(copy.deepcopy(list_transforms))
    if configs['trainable_densenet']:
        if configs['use_horizontal_flip']:
            if configs['prevent_horizontal_flip_in_lateral']:
                train_list_transforms[0] += [image_preprocessing.RandomHorizontalFlipNumpy()]
            else:
                for a_list_transform in train_list_transforms:
                    a_list_transform += [image_preprocessing.RandomHorizontalFlipNumpy()]
        if configs['gamma_range_augmentation'] is not None:
            for a_list_transform in train_list_transforms:
                a_list_transform += [image_preprocessing.RandomGammaAugmentationNumpy(configs['gamma_range_augmentation'])]
        if configs['degree_range_augmentation'] is not None:
            for a_list_transform in train_list_transforms:
                a_list_transform += [image_preprocessing.RandomRotationAugmentationNumpy(configs['degree_range_augmentation'])]
        if configs['scale_range_augmentation'] is not None:
            for a_list_transform in train_list_transforms:
                a_list_transform += [image_preprocessing.RandomScaleAugmentationNumpy(configs['scale_range_augmentation'])]
        if (not configs['use_random_crops']) or ( not (configs['magnification_input']==1 or configs['magnification_input']==4)):
            if configs['chexnet_architecture']=='inception':
                resize_size = 299
            else:
                resize_size = 224
            if configs['use_random_crops']:
                resize_size = int(round(256/224.*resize_size))
            resize_size = int(round(configs['magnification_input']*resize_size))
            for a_list_transform in train_list_transforms:
                a_list_transform += [image_preprocessing.ResizeNumpy(resize_size)]
            for a_list_transform in test_list_transforms:
                a_list_transform += [image_preprocessing.ResizeNumpy(resize_size)]

        if configs['use_random_crops']:
            if configs['chexnet_architecture']=='inception':
                crop_size = 299
            else:
                crop_size = 224
            crop_size = int(round(configs['magnification_input']*crop_size))
            for a_list_transform in train_list_transforms:
                a_list_transform += [image_preprocessing.RandomCropNumpy(crop_size)]
            for a_list_transform in test_list_transforms:
                a_list_transform += [image_preprocessing.CenterCropNumpy(crop_size)]

        if configs['use_half_lung']:
            train_list_transforms[0] += [image_preprocessing.CropOneSideNumpy(0)]
            test_list_transforms[0] += [image_preprocessing.CropOneSideNumpy(0)]
        if not configs['histogram_equalization'] == 'none':
            for a_list_transform in train_list_transforms:
                a_list_transform += [image_preprocessing.HistogramEqualization(configs['normalization_mean'], configs['normalization_std'], local = (configs['histogram_equalization']=='local'))]
            for a_list_transform in test_list_transforms:
                a_list_transform += [image_preprocessing.HistogramEqualization(configs['normalization_mean'], configs['normalization_std'], local = (configs['histogram_equalization']=='local'))]
    for a_list_transform in train_list_transforms:
        a_list_transform += [image_preprocessing.castTensor()]
    for a_list_transform in test_list_transforms:
        a_list_transform += [image_preprocessing.castTensor()]
    trainTransformSequence = [transforms.Compose(a_list_transform) for a_list_transform in train_list_transforms]
    testTransformSequence = [transforms.Compose(a_list_transform)  for a_list_transform in test_list_transforms]
    return sets_of_images, trainTransformSequence, testTransformSequence, num_ftrs
