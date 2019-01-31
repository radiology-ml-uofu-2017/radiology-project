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

def get_name_h5_file():
    return get_name_pickle_file()[:-3]+'h5'
    #return 'all_images_224_224_prot2.2.h5'

def get_name_pickle_file():
    if configs['trainable_densenet']:
        size_all_images = '256_256'
        # if configs['use_random_crops']:
        #     size_all_images = '256_256'
        # else:
        #     size_all_images = '224_224'
    else:
        size_all_images = '7_7'
    return configs['local_directory'] + '/images_' +''.join(configs['data_to_use'])+'_' +  size_all_images + '_prot3.pkl'

class DatasetGenerator(Dataset):
    def __init__ (self, pathDatasetFile, transform = None, train = False, segmentation_features_file = None):
        super(DatasetGenerator, self).__init__()
        self.listImage = pathDatasetFile
        if configs['trainable_densenet'] and configs['load_image_features_from_file']:
            if not configs['machine_to_use'] == 'titan':
                folder = configs['local_directory']
            else:
                folder = '/scratch_ssd/ricbl/pft'
            if not configs['magnification_input']>1:
                self.file = h5py.File(get_name_h5_file(), 'r')
            else:
                assert(configs['magnification_input']<=4)
                self.file = h5py.File(folder + '/images_2012-20162017_1024_1024.prot3.h5', 'r') #'/scratch_ssd/ricbl/pft/images_2012-20162017_1024_1024.prot3.h5', 'r')

            #much slower
            #store = Store('all_images_224_224_prot2.2.h5df', mode="r")
            #self.df1 = store["/frames/1"]
        self.n_images = len(self.listImage[0])
        self.transform = transform
        self.train = train
        if (configs['use_unet_segmentation'] or configs['register_with_segmentation'] or  configs['calculate_segmentation_features']) and configs['segmentation_in_loading']:
            if configs['use_precalculated_segmentation_features']:
                self.segmentation = h5py.File(segmentation_features_file, 'r')
            else:
                self.segmentation = model_loading.SegmentationNetwork().cuda()
    #--------------------------------------------------------------------------------

    def __getitem__(self, index):
        if not isinstance(index, int):
            index = int(index)
        imageData = []
        filepath=[]
        vectorized_features = 0
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

                    imageData.append(-self.file['dataset_1'][old_index,...].astype('float32')) #TEMP: corrrection because input is inversed

                    #much slower:
                    #examples_to_read = ['ind' + str(old_index)]
                    #imageData = self.df1.rows(examples_to_read).values.reshape(3,224,224)
                else:
                    imageData.append(self.listImage[i]['preprocessed'].iloc[index][0])

            if self.transform[i] is not None:
                imageData[i] = self.transform[i](imageData[i])
                #torchvision.utils.save_image(imageData[i] , './test_transform_cut' + str(i)+'.png')
                '''
                if i == 0:
                    imageData[i] = image_preprocessing.CropOneSideNumpy(None)(imageData[i])
                else:
                    imageData[i] = image_preprocessing.CropSideCenterNumpy(None)(imageData[i])
                '''

            #imageData[i] = np.expand_dims(imageData[i], 0)
            filepath.append(self.listImage[i]['filepath'].iloc[index])

            if (configs['use_unet_segmentation'] or configs['register_with_segmentation'] or configs['calculate_segmentation_features']) and configs['segmentation_in_loading'] and not configs['use_precalculated_segmentation_features']:
                #assert(not configs['register_with_segmentation']) #TODO: implement registration somewhere
                if i == 0 or configs['use_unet_segmentation_for_lateral']:
                    imageData[i], features = self.segmentation(imageData[i].cuda().unsqueeze(0))
                    
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
        if (configs['relative_noise_to_add_to_label'] is not None) and (self.train):
            imageLabel = torch.FloatTensor(np.random.normal(imageLabel, imageLabel * configs['relative_noise_to_add_to_label']))
        imageAllColumns= torch.FloatTensor(self.listImage[0][configs['all_output_columns']].iloc[index])
        if configs['use_extra_inputs']:
            col_extra_inputs = [col for col in self.listImage[0] if col.startswith('tobacco_status_')]+[col for col in self.listImage[0] if col.startswith('smoking_tobacco_status_')]+['binary_gender']+['age']
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
        return imageData[0], ([] if len(self.listImage)==1 else imageData[1]), imageLabel, imageAllColumns, extra_inputs, filepath, vectorized_features, index

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
        #torchvision.utils.save_image((frontal_images+2.5)/5, 'testchestxray14'+str(index)+'.png')
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

        # h5f = h5py.File('/scratch_ssd/ricbl/pft/images_2012-20162017_1024_1024.prot3.h5', 'w')
        # h5f.create_dataset('dataset_1', (n_images, 3, 1024, 1024))
        # all_images = image_preprocessing.preprocess_images_and_save(all_images, transformSequence, h5f)
        # h5f.close()

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

        if configs['use_lateral']:
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
    if configs['use_lateral']:
        train_list_transforms.append(copy.deepcopy(list_transforms))
        test_list_transforms.append(copy.deepcopy(list_transforms))
    if configs['trainable_densenet']:
        if configs['use_horizontal_flip']:
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
        if (not configs['use_random_crops']) or ( not (configs['magnification_input']==1)):
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
