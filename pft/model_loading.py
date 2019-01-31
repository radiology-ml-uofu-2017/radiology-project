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
import utils
import math
import unet_models
from torch.nn.modules.utils import _pair
from scipy.stats import multivariate_normal
import image_preprocessing
from skimage.measure import label,regionprops, perimeter
import scipy
import time
from skimage.feature import corner_harris, corner_subpix, corner_peaks, corner_shi_tomasi, corner_foerstner,corner_fast, corner_orientations
import matplotlib.pyplot as plt
from skimage.morphology import octagon, disk, binary_erosion, watershed
from skimage.draw import ellipse, ellipse_perimeter, polygon, line
import copy
from collections import OrderedDict
from scipy.optimize import curve_fit

np.seterr(divide = "raise", over="warn", under="warn",  invalid="raise")

labels_columns = configs['get_labels_columns']

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
                if res and utils.compare_versions(torchvision.__version__, '0.2.1')>=0:
                    new_key = res.group(1) + res.group(2)
                new_key = new_key.replace('densenet121', 'model')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model

def load_pretrained_chexnet():
    chexnetModel = CheXNet(14, configs['chexnet_layers'], configs['chexnet_architecture']).cuda()
    chexnetModel = torch.nn.DataParallel(chexnetModel).cuda()
    if configs['pretrain_kind']=='chestxray':
        chexnetModel = load_checkpoint(chexnetModel)
    if configs['use_local_conv']:
        chexnetModel.module.model.layer4.add_module('conv_local_1', Conv2dLocal(in_height= configs['avg_pool_kernel_size'], in_width= configs['avg_pool_kernel_size'], in_channels=512, out_channels = configs['n_channels_local_convolution'],
                 kernel_size= (3,3), stride=1, padding=1, bias=True, dilation=1))
        #chexnetModel.module.model.layer4.add_module('relu_local_conv', nn.ReLU(inplace= True))
        #chexnetModel.module.model.layer4.add_module('conv_local_2',Conv2dLocal(in_height= configs['avg_pool_kernel_size'], in_width= configs['avg_pool_kernel_size'], in_channels=configs['n_channels_local_convolution'], out_channels = configs['n_channels_local_convolution'],
        #         kernel_size= (3,3), stride=1, padding=1, bias=True, dilation=1))
        #chexnetModel.layer4.1.bn1 = nn.Dropout2d(p=0.25)
        #chexnetModel.module.model.layer4[1].bn1 = nn.Sequential()
        #chexnetModel.layer4.1.bn2 = nn.Dropout2d(p=0.25)
        #chexnetModel.module.model.layer4[1].bn2 = nn.Sequential()
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

        if configs['fully_connected_kind'] == 'fully_connected':
            self.fc_part = ModelFCPart(self.spatial_part.current_n_channels)
        elif configs['fully_connected_kind'] == 'dsnm':
            self.fc_part = ModelDSNMPart(self.spatial_part.current_n_channels)
        elif configs['fully_connected_kind'] == 'softmax_gate':
            self.fc_part = ModelInternalClassSelection(self.spatial_part.current_n_channels)
        print(self.fc_part.current_n_channels)
        self.final_linear = ModelLastLinearLayer(self.fc_part.current_n_channels)#527)

    def forward(self, x, extra_fc_input, epoch ):
        x, spatial_outputs = self.spatial_part(x)
        averaged_spatial_output = x
        #x, ws = self.fc_part(x, extra_fc_input, epoch)
        x, extra_outs = self.fc_part(x, extra_fc_input)
        x, extra_outs2 = self.final_linear(x)
        dict_out = utils.merge_two_dicts(extra_outs,extra_outs2)
        dict_out['spatial_outputs'] = spatial_outputs
        dict_out['averaged_spatial_output'] = averaged_spatial_output
        return x, dict_out

class SegmentationNetwork(nn.Module):
    def __init__(self):
        super(SegmentationNetwork, self).__init__()

        self.segmentation = self.get_pretrained_unet()

        self.padding_margin = 4*8

    def load_unet_model(self, model):
        model.load_state_dict(torch.load(configs['local_directory'] + '/models/'+configs['unet_model_file']))
        return model

    def get_pretrained_unet(self):
        if configs['use_unet_segmentation'] or configs['register_with_segmentation'] or configs['calculate_segmentation_features']:
            model = unet_models.UNet()
            model = self.load_unet_model(model)
            return model
        else:
            return nn.Sequential()

    def get_features_for_region(self, region_properties_image, segmented_xi, xi_to_segment, region_in_the_left_of_image, originally_right_lung, ax, i):
        image_corners = np.pad(segmented_xi.squeeze(0).cpu().numpy(), ((self.padding_margin,self.padding_margin),(self.padding_margin,self.padding_margin)), mode = 'constant')
        original_image_padded = np.pad(xi_to_segment.cpu().numpy(), ((0,0),(self.padding_margin,self.padding_margin),(self.padding_margin,self.padding_margin)), mode = 'constant')
        chosen_corner = self.get_two_corners_of_interest(region_in_the_left_of_image, originally_right_lung, image_corners)
        features = None
        image_corners2 = copy.copy(image_corners)
        if len(chosen_corner)>1:
            features = self.get_features_lung(region_properties_image, chosen_corner, image_corners, self.padding_margin, region_in_the_left_of_image, original_image_padded)
            if ax is not None:
                ax[i].imshow(image_corners, interpolation='nearest', cmap=plt.cm.gray)
        if ax is not None:
            ax[i].plot([corner[1] for corner in chosen_corner], [corner[0] for corner in chosen_corner], 'xg', markersize=15)
        return features

    def get_two_corners_of_interest(self, region_in_the_left_of_image, originally_right_lung, image_corners):
        sigma = (8 if originally_right_lung else 4)
        coords = corner_peaks(corner_harris(image_corners, sigma = sigma, k = 0.01), min_distance=(10 if originally_right_lung else 2), num_peaks = (5 if originally_right_lung else 10))
        coords = sorted(coords, key = lambda x: 0.45*x[1]+x[0] if not region_in_the_left_of_image else -0.45*x[1]+x[0])
        chosen_corner = []
        corner_mask = disk(2*sigma)
        for corner in reversed(coords):
            corner_s = corner_subpix(image_corners, np.expand_dims(corner, axis = 0), window_size=30)[0]
            corner_s_not_found = np.isnan(corner_s[0])
            corner_s = corner  if corner_s_not_found else corner_s.astype(np.int)
            corner = corner.astype(np.int)
            angle = np.rad2deg(corner_orientations(image_corners, np.expand_dims(corner_s, axis = 0), corner_mask))[0]
            intensity = (np.mean(image_corners[corner_s[0]-sigma*3,corner_s[1]-sigma*3:corner_s[1]+sigma*3-1])+ \
                        np.mean(image_corners[corner_s[0]-sigma*3:corner_s[0]+sigma*3-1,corner_s[1]+sigma*3]) + \
                        np.mean(image_corners[corner_s[0]+sigma*3,corner_s[1]-sigma*3+1:corner_s[1]+sigma*3]) + \
                        np.mean(image_corners[corner_s[0]-sigma*3+1:corner_s[0]+sigma*3,corner_s[1]-sigma*3]))/4.
            if angle > 90:
                angle = -360+angle
            if len(chosen_corner)==0:
                if angle<-25 \
                and angle>-155:
                    chosen_corner += [corner]
            elif len(chosen_corner)==1:
                if ((intensity<0.5) if not corner_s_not_found else (image_corners[corner_s[0]+1,corner_s[1]]==1)) \
                and angle<(-70 if region_in_the_left_of_image else 20) \
                and angle>(-200 if region_in_the_left_of_image else -110) \
                and ((corner[1]>chosen_corner[0][1]) if region_in_the_left_of_image else (corner[1]<chosen_corner[0][1])) \
                and -corner[0]+chosen_corner[0][0]<60 \
                and abs(corner[1]-chosen_corner[0][1])>0.7*abs(-corner[0]+chosen_corner[0][0]):
                    chosen_corner += [corner]
                    break
        chosen_corner_s = corner_subpix(image_corners, np.array(chosen_corner), window_size=30)
        chosen_corner = [chosen_corner_s[i1].astype(np.int) if not np.isnan(chosen_corner_s[i1][0]) else chosen_corner[i1].astype(np.int) for i1 in range(len(chosen_corner))]
        if len(chosen_corner)==2 and (not ((chosen_corner[1][1]>chosen_corner[0][1]) if region_in_the_left_of_image else (chosen_corner[1][1]<chosen_corner[0][1]))):
            chosen_corner = chosen_corner[:1]
        return chosen_corner

    def get_features_lung(self, region_properties_image, chosen_corner, image_corners, padding_margin, region_in_the_left_of_image, original_image_padded):
        features = OrderedDict()
        features.update({'row_corner_0':chosen_corner[0][0]-padding_margin,
                    'column_corner_0':chosen_corner[0][1]-padding_margin,
                    'row_corner_1':chosen_corner[1][0]-padding_margin,
                    'column_corner_1':chosen_corner[1][1]-padding_margin})
        try:
            features['angle'] = np.arctan((chosen_corner[0][0]-chosen_corner[1][0])/float(chosen_corner[1][1]-chosen_corner[0][1])) if region_in_the_left_of_image else np.arctan((chosen_corner[1][0]-chosen_corner[0][0])/float(chosen_corner[0][1]-chosen_corner[1][1]))
        except FloatingPointError:
            print(chosen_corner[1][1]-chosen_corner[0][1])
            print(chosen_corner)
            raise
        features['distance'] = math.sqrt((chosen_corner[0][0]-chosen_corner[1][0])**2+(chosen_corner[0][1]-chosen_corner[1][1])**2)
        features['area_lung'] = region_properties_image.area
        region_list_coords = region_properties_image.coords+padding_margin

        features.update(self.get_diaphragm_measures(chosen_corner, features['distance'], features['angle'], image_corners, region_in_the_left_of_image))
        features.update(self.get_upper_void_scores(image_corners, region_list_coords, features['area_lung'], region_in_the_left_of_image))

        features['orientation_lung'] = region_properties_image.orientation
        features['lung_elongation'] = region_properties_image.inertia_tensor_eigvals[0]/region_properties_image.inertia_tensor_eigvals[1]
        features['bounding_box_occupation'] = region_properties_image.extent
        features['lung_eccentricity'] = region_properties_image.eccentricity
        features['average_intensity_lung'] = np.sum(np.expand_dims(image_corners, axis = 0)*original_image_padded)/float(np.sum(image_corners))
        #TODO: maybe normalize by intensity of spine
        #TODO: how to deal with devices inside the lung
        centroid_row = int(round(region_properties_image.centroid[0]+padding_margin))
        features['average_upper_half_intensity_lung'] = np.sum(np.expand_dims(image_corners[:centroid_row,:], axis = 0)*original_image_padded[:,:centroid_row,:])/float(np.sum(image_corners[:centroid_row,:])) #TODO: check if this is working
        features['temp_mult'] = (np.expand_dims(image_corners[:centroid_row,:], axis = 0)*original_image_padded[:,:centroid_row,:])/2+0.5
        return features

    def get_diaphragm_measures(self, chosen_corner, distance, angle, image_corners, region_in_the_left_of_image):
        watershed_score = self.get_diaphragm_watershed_score(chosen_corner, image_corners, distance, region_in_the_left_of_image)
        try:
            rr, cc = ellipse(int((chosen_corner[0][0]+chosen_corner[1][0])/2), int((chosen_corner[0][1]+chosen_corner[1][1])/2),max(int(round(0.3*distance)),10), int(round(distance/2.)), rotation=angle, shape = image_corners.shape)
            a = np.zeros_like(image_corners)
            a[rr, cc] = 1
            only_ellipse_perimeter = a-binary_erosion(a)
            masked = a*image_corners
            masked[np.nonzero(only_ellipse_perimeter)] = 1
            if distance>=5:
                try:
                    curl_diaphragm = float(distance)/(perimeter(masked, neighbourhood  = 8)-perimeter(a, neighbourhood  = 8)/2.-perimeter(only_ellipse_perimeter, neighbourhood  = 8)/2.)
                except FloatingPointError:
                    print(perimeter(only_ellipse_perimeter, neighbourhood  = 8))
                    print(perimeter(masked, neighbourhood  = 8))
                    print(perimeter(a, neighbourhood  = 8))
                    print(distance)
                    curl_diaphragm = 0
            else:
                curl_diaphragm = 1
            ellipse_average_intensity = np.sum(image_corners[rr, cc])/float(len(rr))
        except FloatingPointError:
            print(int((chosen_corner[0][0]+chosen_corner[1][0])/2))
            print( int((chosen_corner[0][1]+chosen_corner[1][1])/2))
            print(distance)
            curl_diaphragm = 0
            ellipse_average_intensity = 0
        return OrderedDict({'curl_diaphragm': curl_diaphragm,
                'ellipse_average_intensity': ellipse_average_intensity,
                'watershed_diaphragm_score':watershed_score})

    def get_diaphragm_watershed_score(self, chosen_corner, image_corners, distance, region_in_the_left_of_image):
        upper_polygon = polygon([chosen_corner[0][0],chosen_corner[1][0], 0, 0], [chosen_corner[0][1],chosen_corner[1][1],chosen_corner[1][1], chosen_corner[0][1]], shape = image_corners.shape )
        image_white_pixels_under = copy.copy(image_corners)
        image_white_pixels_under[upper_polygon] = 0.0
        white_pixels_under = np.sum(image_white_pixels_under[:,chosen_corner[0][1]:chosen_corner[1][1]]) if region_in_the_left_of_image else np.sum(image_white_pixels_under[:,chosen_corner[1][1]:chosen_corner[0][1]])
        under_polygon = polygon([chosen_corner[0][0],chosen_corner[1][0], 3000, 3000], [chosen_corner[0][1],chosen_corner[1][1],chosen_corner[1][1], chosen_corner[0][1]], shape = image_corners.shape )
        image_black_pixels_over = copy.copy(image_corners)
        image_black_pixels_over[under_polygon] = 1.0
        if region_in_the_left_of_image:
            image_black_pixels_over[:,:chosen_corner[0][1]+1] = 1.0
        else:
            image_black_pixels_over[:,:chosen_corner[1][1]+1] = 1.0
        if region_in_the_left_of_image:
            image_black_pixels_over[:,chosen_corner[1][1]:] = 1.0
        else:
            image_black_pixels_over[:,chosen_corner[0][1]:] = 1.0
        dividing_line = line(chosen_corner[0][0], chosen_corner[0][1], chosen_corner[1][0], chosen_corner[1][1])
        dividing_line = list(dividing_line)
        dividing_line[0] = dividing_line[0][1:-1]
        dividing_line[1] = dividing_line[1][1:-1]
        markers_watershed = np.zeros_like(image_corners)
        markers_watershed[dividing_line] = 1.0
        watershed_black_pixels_over = watershed(1-image_black_pixels_over,markers_watershed,mask = 1-image_black_pixels_over)
        black_pixels_over = np.sum(watershed_black_pixels_over) - np.sum(markers_watershed)
        white_pixels_on_line = np.sum(markers_watershed*image_corners)
        watershed_score = (black_pixels_over-white_pixels_on_line-white_pixels_under)/float(distance)
        return watershed_score

    def get_upper_void_scores(self, image_corners, region_list_coords, area_lung, region_in_the_left_of_image):
        first_point_region = region_list_coords[np.argmin(region_list_coords[:,0]),:]
        lower_corner_region = np.array([np.amax(region_list_coords[:,0]), np.amin(region_list_coords[:,1]) if region_in_the_left_of_image else np.amax(region_list_coords[:,1])])
        image_convexity_upper_void = copy.copy(image_corners)
        image_convexity_upper_void[:first_point_region[0], :] = 1
        image_convexity_upper_void[lower_corner_region[0]-1:, :] = 1
        if region_in_the_left_of_image:
            image_convexity_upper_void[:, :lower_corner_region[1]] = 1
            image_convexity_upper_void[:, first_point_region[1]-1:] = 1
        else:
            image_convexity_upper_void[:, lower_corner_region[1]+1:] = 1
            image_convexity_upper_void[:, :first_point_region[1]+2] = 1
        watershed_void_marker = np.zeros_like(image_convexity_upper_void)
        watershed_void_marker[first_point_region[0],lower_corner_region[1]] = 1.0
        watershed_void = watershed(1-image_convexity_upper_void,watershed_void_marker,mask = 1-image_convexity_upper_void)
        bounding_box_upper_corner_void_occupation = float(np.sum(watershed_void))/(area_lung/2.)
        upper_corner_void_convexity = float(np.sum(watershed_void))/float(abs(first_point_region[0]-lower_corner_region[0])*abs(first_point_region[1]-lower_corner_region[1])/2.)
        return OrderedDict({'bounding_box_upper_corner_void_occupation':bounding_box_upper_corner_void_occupation,
        'upper_corner_void_convexity':upper_corner_void_convexity,
        'temp_watershed':watershed_void})

    def get_theta(self, regions_properties_image, rows = 224, cols = 224):
        if len(regions_properties_image)>1:
            bounding_box = [min(regions_properties_image[0].bbox[0],regions_properties_image[1].bbox[0]), \
                            min(regions_properties_image[0].bbox[1],regions_properties_image[1].bbox[1]), \
                            max(regions_properties_image[0].bbox[2],regions_properties_image[1].bbox[2]), \
                            max(regions_properties_image[0].bbox[3],regions_properties_image[1].bbox[3])]
        else:
            bounding_box = regions_properties_image[0].bbox
        '''
        bounding_box = [max(bounding_box[0]-int(rows/224.*15),0), \
                        max(bounding_box[1]-int(cols/224.*15),0), \
                        min(bounding_box[2]+int(rows/224.*15),rows), \
                        min(bounding_box[3]+int(cols/224.*15),cols)]
        '''
        theta = torch.tensor(
        [[1./(-float(cols)/(bounding_box[1]-bounding_box[3])), \
        0, \
        bounding_box[1]*2./(float(cols))-1+1./(-float(cols)/(bounding_box[1]-bounding_box[3]))], \
        [0, \
        1./(-float(rows)/(bounding_box[0]-bounding_box[2])), \
        bounding_box[0]*2./(float(rows))-1+1./(-float(rows)/(bounding_box[0]-bounding_box[2]))] \
        ]).float().cuda()
        return theta

    def get_segmentation_and_features(self, xi_to_segment):
        if configs['extra_histogram_equalization_for_segmentation']:
            assert(configs['initial_lr_unet']==0)
            xi_to_segment = torch.tensor(np.array([image_preprocessing.HistogramEqualization(configs['normalization_mean'], configs['normalization_std'])(image) for image in xi_to_segment.detach().cpu().numpy()])).float().cuda()
        segmented_xi = 1-torch.sigmoid(self.segmentation(torch.mean(xi_to_segment, dim = 1, keepdim = True)))
        if not (configs['magnification_input']==1):
            segmented_xi = torch.nn.functional.interpolate(segmented_xi, scale_factor=configs['magnification_input'], mode = 'bilinear')
        if configs['initial_lr_unet']==0:
            segmented_xi = segmented_xi.detach()
        # torchvision.utils.save_image( segmented_xi, './test_segmentation.png')
        # torchvision.utils.save_image( 1-torch.mean(image_preprocessing.BatchUnNormalizeTensor([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])(xi), dim = 1, keepdim = True), './test_segmentation_origin.png')
        # 1/0
        features_images = None
        if configs['register_with_segmentation'] or configs['calculate_segmentation_features']:
            assert(configs['initial_lr_unet']==0)
            segmented_xi = segmented_xi>0.5
            labeled_images = [(label(image)) for image in segmented_xi.detach().cpu().numpy().astype(np.int)]
            ax = None
            # fig, ax = plt.subplots(1,13,figsize=(2.24*13, 2.24))
            features_images = []
            regions_properties_images = []
            for k in range(len(labeled_images)):
                labeled_image = labeled_images[k]
                labeled_image[:,segmented_xi.size(3):] *= (np.amax(labeled_image)+1)
                labeled_images[k] = labeled_image
                regions_properties_images.append(regionprops(labeled_image))
            segmented_xis = torch.zeros_like(segmented_xi).float().cuda()
            #segmented_xi = torch.tensor(np.array([((labeled_images[k] == biggest_regions[k][0])+(labeled_images[k] == biggest_regions[k][1])) for k in range(len(biggest_regions) )]).astype(np.float32)).cuda()
            for k in range(len(labeled_images)):
                labeled_image = labeled_images[k]
                regions_properties_image = regions_properties_images[k]
                #biggest_regions = [np.argpartition([region.area for region in regions_properties_image], -2)[-2:]+1 for regions_properties_image in regions_properties_images]
                if len(regions_properties_image)>=2:
                    biggest_regions = np.argpartition([region.area for region in regions_properties_image], -2)[-2:]+1
                    regions_properties_image  = [regions_properties_image[biggest_regions[0]-1],regions_properties_image[biggest_regions[1]-1]]
                else:
                    biggest_regions = [1]
                segmented_xi_list = []
                for biggest_region in biggest_regions:
                    segmented_xi_region = torch.tensor(np.array(labeled_image == biggest_region).astype(np.float32)).cuda()
                    segmented_xi_list.append(segmented_xi_region)
                    segmented_xis[k,...] += segmented_xi_region
                features_regions = {}
                if not configs['use_horizontal_flip']:
                    original_lung_position_in_body = []
                    for j in range(len(regions_properties_image)):
                        original_lung_position_in_body.append('right' if (regions_properties_image[j].centroid[1]<segmented_xi.size(3)/2) else 'left')
                else:
                    if len(regions_properties_image)>1:
                        first_big = regions_properties_image[0].area>regions_properties_image[1].area
                        first_elongated = regions_properties_image[0].inertia_tensor_eigvals[0]/regions_properties_image[0].inertia_tensor_eigvals[1] - regions_properties_image[1].inertia_tensor_eigvals[0]/regions_properties_image[1].inertia_tensor_eigvals[1]
                        original_lung_position_in_body = ['right', 'left'] if (first_elongated < 0 if abs(first_elongated)>0.5 else first_big) else ['left', 'right']
                    else:
                        original_lung_position_in_body = ['left']
                for j in range(len(segmented_xi_list)):
                    originally_right_lung = original_lung_position_in_body[j]=='right'
                    region_in_the_left_of_image = regions_properties_image[j].centroid[1]<segmented_xi.size(3)/2
                    if region_in_the_left_of_image and not originally_right_lung:
                        assert(False)
                    features = self.get_features_for_region( regions_properties_image[j], segmented_xi_list[j],xi_to_segment[k],region_in_the_left_of_image, originally_right_lung, ax, k)
                    features_regions[original_lung_position_in_body[j]] = features
                features_regions['theta'] = self.get_theta(regions_properties_image, segmented_xi.size(2), segmented_xi.size(3))
                features_images.append(features_regions)
            # fig.savefig('./test_corners.png')
            # plt.close('all')
        else:
            segmented_xis = segmented_xi
        return segmented_xis, features_images

    def forward(self, xi):
        if not (configs['magnification_input']==1):
            xi_to_segment = torch.nn.functional.interpolate(xi, scale_factor=1./configs['magnification_input'], mode = 'bilinear')
        else:
            xi_to_segment = xi
        xi_to_segment = -((image_preprocessing.BatchUnNormalizeTensor(configs['normalization_mean'],configs['normalization_std'])(xi_to_segment))-0.5)*2

        segmented_xi, features = self.get_segmentation_and_features(xi_to_segment)
        if configs['register_with_segmentation']:
            grid = F.affine_grid(torch.cat([features_regions['theta'].unsqueeze(0) for features_regions in features]), segmented_xi.size())
            segmented_xi = F.grid_sample(segmented_xi, grid)
            xi = F.grid_sample(xi, grid)

        if configs['use_unet_segmentation']:
            mult_xi = xi*segmented_xi
            if configs['unet_multiply_instead_of_channel']:
                xi = mult_xi
            else:
                segmented_xi = segmented_xi.expand(-1,3,-1,-1)
                if configs['normalization_segmentation']:
                    segmented_xi = image_preprocessing.BatchNormalizeTensor(configs['normalization_mean'],configs['normalization_std'])(segmented_xi)
                xi = torch.cat([xi[:,0:1,...], mult_xi[:,0:1,...], segmented_xi[:,2:3,...]], dim = 1)
        return xi, features

class ModelMoreInputs(nn.Module):
    def __init__(self, num_ftrs):
        super(ModelMoreInputs, self).__init__()

        self.n_inputs, self.n_cnns = get_qt_inputs()
        self.stn = STN()
        if (configs['use_unet_segmentation'] or configs['register_with_segmentation'] or configs['calculate_segmentation_features']) and not configs['segmentation_in_loading']:
            self.segmentation = SegmentationNetwork()
        else:
            self.segmentation =nn.Sequential()

        cnns = []
        bns = []
        for i in range(self.n_cnns):
            if configs['trainable_densenet']:
                new_cnn = get_model_cnn()
                num_ftrs = new_cnn.num_ftrs
                cnns.append(new_cnn)

            else:
                cnns.append(nn.Sequential())
            if configs['normalize_lateral_and_frontal_with_bn']:
                bns.append(torch.nn.BatchNorm2d(num_ftrs))
        self.cnn = nn.ModuleList(cnns)
        self.bns = nn.ModuleList(bns)
        self.final_layers = FinalLayers(num_ftrs, self.n_inputs)

    def forward(self, images1, images2, extra_fc_input, epoch ):
        all_outputs = []
        imageList = [images1, images2]
        for i in range(self.n_inputs):
            if configs['tie_cnns_same_weights']:
                index = 0
            else:
                index = i
            if configs['use_spatial_transformer_network']:
                xi = self.stn(imageList[i])
            else:
                xi = imageList[i]
            if (configs['use_unet_segmentation'] or configs['register_with_segmentation'] or configs['calculate_segmentation_features']) and not configs['segmentation_in_loading']:
                if i == 0 or configs['use_unet_segmentation_for_lateral']:
                    xi, _ = self.segmentation(xi)
            xi = self.cnn[index](xi)
            if configs['normalize_lateral_and_frontal_with_bn']:
                xi = self.bns[index](xi)
            all_outputs.append(xi)
        x, extra_outs = self.final_layers(all_outputs, extra_fc_input, epoch)
        return x, extra_outs

def my_eye_(tensor):
    with torch.no_grad():
        tensor = torch.eye(2,3, requires_grad=tensor.requires_grad).view(tensor.size(0))
    return tensor

class LocationNetwork(nn.Module):
    def __init__(self, w_h):
        super(LocationNetwork, self).__init__()
        self.current_n_channels = 3
        self.location_part = torch.nn.Sequential()

        '''
        for layer_i in range(int(math.log(w_h/7., 2))):
            if configs['use_batchnormalization_location']:
                self.location_part.add_module("location_bn_"+str(layer_i),torch.nn.BatchNorm2d(self.current_n_channels).cuda())
            self.location_part.add_module("location_drop_"+str(layer_i),nn.Dropout(p=configs['use_dropout_location']).cuda())
            self.location_part.add_module("location_conv_"+str(layer_i),nn.Conv2d(self.current_n_channels, configs['channels_location'], kernel_size = 5, stride=1, padding=2, dilation=1, groups=1, bias=True))
            self.current_n_channels = configs['channels_location']
            self.location_part.add_module("location_nonlinearity_"+str(layer_i),nn.ReLU().cuda())
            self.location_part.add_module("location_pool_"+str(layer_i),nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False))
        #current_model.add_module("pool_"+str(layer_i),nn.AvgPool2d(kernel_size=7, stride=7, padding=0, ceil_mode=False, count_include_pad=True))
        self.location_part.add_module("location_flatten",Flatten().cuda())
        self.current_n_channels = self.current_n_channels*7*7
        self.location_part.add_module("location_linear_0",nn.Linear(self.current_n_channels, configs['channels_location'] ).cuda())
        self.current_n_channels = configs['channels_location']
        self.location_part.add_module("location_fc_nonlinearity_0",nn.ReLU().cuda())
        self.location_part.add_module("location_linear_1",nn.Linear(self.current_n_channels, configs['channels_location'] ).cuda())
        self.current_n_channels = configs['channels_location']
        self.location_part.add_module("location_fc_nonlinearity_1",nn.ReLU().cuda())
        self.linear_out = nn.Linear(self.current_n_channels, 6).cuda()
        '''
        '''
        self.location_part.add_module("location_conv_0",nn.Conv2d(3, 32, kernel_size = 5, stride=1, padding=2,bias=True))
        self.location_part.add_module("location_nonlinearity_0",nn.ReLU().cuda())
        self.location_part.add_module("location_pool_0",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False))
        self.location_part.add_module("location_conv_1",nn.Conv2d(32, 32, kernel_size = 5, stride=1, padding=2,bias=True))
        self.location_part.add_module("location_nonlinearity_1",nn.ReLU().cuda())
        self.location_part.add_module("location_flatten",Flatten().cuda())
        self.location_part.add_module("location_linear_0",nn.Linear(16*16*32, 32 ).cuda())
        self.location_part.add_module("location_fc_nonlinearity_0",nn.ReLU().cuda())
        self.location_part.add_module("location_linear_1",nn.Linear(32, 32 ).cuda())
        self.location_part.add_module("location_fc_nonlinearity_1",nn.ReLU().cuda())
        self.linear_out = nn.Linear(32, 6).cuda()
        '''

        self.location_part.add_module("location_conv_0",nn.Conv2d(3, 18, kernel_size = 5, stride=1, padding=0,bias=True))
        self.location_part.add_module("location_nonlinearity_0",nn.ReLU().cuda())
        self.location_part.add_module("location_pool_0",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False))
        self.location_part.add_module("location_conv_1",nn.Conv2d(18, 16*3, kernel_size = 5, stride=1, padding=0,bias=True))
        self.location_part.add_module("location_nonlinearity_1",nn.ReLU().cuda())
        self.location_part.add_module("location_pool_1",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False))
        self.location_part.add_module("location_flatten",Flatten().cuda())
        self.location_part.add_module("location_linear_0",nn.Linear(5*5*16*3, 120 ).cuda())
        self.location_part.add_module("location_fc_nonlinearity_0",nn.ReLU().cuda())
        self.location_part.add_module("location_linear_1",nn.Linear(120, 55 ).cuda())
        self.location_part.add_module("location_fc_nonlinearity_1",nn.ReLU().cuda())
        self.linear_out = nn.Linear(55, 6).cuda()

        torch.nn.init.constant_(self.linear_out.weight, 0.0)
        self.linear_out.bias = torch.nn.Parameter(my_eye_(self.linear_out.bias))
        '''
        self.linear_out.weight.data.zero_()
        self.linear_out.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        '''
    def forward(self, x):
        x = utils.downsampling(x, (32,32), None, 'bilinear')
        x = self.location_part(x)
        x = self.linear_out(x)
        return x

class STN(nn.Module):
    def __init__(self, w_h=224):
        super(STN, self).__init__()
        if configs['use_spatial_transformer_network']:
            self.localisation = LocationNetwork(w_h)
        else:
            self.localisation = nn.Sequential()
    def forward(self, x):
        if configs['use_spatial_transformer_network']:
            theta = self.localisation(x)
            theta = theta.view(-1, 2, 3)
            print(theta)
            grid = F.affine_grid(theta, x.size())
            x = F.grid_sample(x, grid)

        return x

class _ConvNdGaussians(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups,7))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, 7))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class Conv2dGaussians(_ConvNdGaussians):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        self.kernel__size = kernel_size
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        x, y = np.mgrid[-(kernel_size-1)/2:(kernel_size-1)/2:1, -(kernel_size-1)/2:(kernel_size-1)/2:1]
        self.pos = np.empty(x.shape + (2,))
        self.pos[:, :, 0] = x; self.pos[:, :, 1] = y

    def get_gaussian_kernel(kernel_size, scale, mean, covariance):
        rv = multivariate_normal(mean, covariance)
        return scale * rv.pdf(pos)

    def forward(self, input):
        return F.conv2d(input, get_gaussian_kernel(self.kernel_size, self.weight[:,:,0], self.weight[:,:,1:3], self.weight[:,:,3:7].view(2,2)) , self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class quickAttention(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x[:,0:1,:,:])*x

def get_spatial_part_fc(num_ftrs, cnn_number):
    spatial_part = torch.nn.Sequential()
    spatial_part.add_module("spatial_output",torch.nn.Sequential())
    current_n_channels = num_ftrs
    print(current_n_channels)
    if configs['use_conv11']:
        if configs['use_batchnormalization_hidden_layers']:
            spatial_part.add_module("bn_conv11",torch.nn.BatchNorm2d(current_n_channels/2, momentum = 0.1).cuda())
        spatial_part.add_module("conv11",nn.Conv2d( in_channels = current_n_channels/2, out_channels =  configs['conv11_channels'], kernel_size = 1).cuda())
        if not configs['use_sigmoid_channel']:
            spatial_part.add_module("reluconv11",nn.ReLU().cuda())
        current_n_channels = configs['conv11_channels']*2
    if configs['use_sigmoid_channel']:
        spatial_part.add_module("sigmoid_channel",quickAttention().cuda())
        spatial_part.add_module("reluconv11",nn.ReLU().cuda())
    if not configs['remove_pre_avg_pool']:
        if not configs['use_delayed_lateral_pooling']:
            kernel_size = configs['avg_pool_kernel_size']
            kernel_size = int(round(configs['magnification_input']*kernel_size))
            if configs['use_half_lung'] and cnn_number==0:
                kernel_size = (kernel_size,kernel_size/2)
            spatial_part.add_module("avgpool",nn.AvgPool2d(kernel_size).cuda())
        else:
            spatial_part.add_module("avgpool",nn.AvgPool2d(kernel_size = (1,configs['avg_pool_kernel_size'])).cuda())
    else:
        current_n_channels = current_n_channels*49

    #spatial_part.add_module("flatten",Flatten().cuda())
    return spatial_part, current_n_channels

class MeanVarLossMean(nn.Module):
    def __init__(self, col_name, bins):
        super(MeanVarLossMean, self).__init__()
        self.bins = bins
    
    def set_bins(self, bins):
        self.bins = bins
    
    def forward(self, input):
        x = torch.sum(input*self.bins.expand_as(input), dim = 1)
        if len(x.size())>1:
            x = x.squeeze(1)
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
            if utils.compare_versions(torch.__version__, '0.4.0')>=0:
                one_output.add_module('logits_to_probs', nn.Softmax(dim = 1))
            else:
                one_output.add_module('logits_to_probs', nn.Softmax())
            one_output.add_module('probs_to_mean', MeanVarLossMean(output_column_name, bins[index]))
            outputs_list.append(one_output)
        self.intermediary_probabilities_module_list = nn.ModuleList(intermediary_probabilities_list)
        self.outputs_module_list = nn.ModuleList(outputs_list)
        
    def set_bins(self, bins, index):
        self.bins[index] = bins
        self.outputs_module_list[index].probs_to_mean.set_bins(bins)
        
    def forward(self, input):
        intermediary_logits_list = []
        outputs_list = []
        size0 = input.size()[0]
        argmax_list = []
        for index, output_column_name in enumerate(configs['get_labels_columns']):
            #TODO: if output kind is sigmoid, or maybe if output is copd, skip next two lines
            intermediary_logits_one_output = self.intermediary_probabilities_module_list[index](input)
            intermediary_logits_list.append(intermediary_logits_one_output)
            one_output = self.outputs_module_list[index](intermediary_logits_one_output)
            outputs_list.append(one_output)
            max_probs = torch.max(intermediary_logits_one_output, dim=1)[1]
            if len(max_probs.size())>1:
                max_probs = max_probs.squeeze(1)
            argmax_list.append(torch.index_select(self.bins[index], dim = 1, index = max_probs).squeeze(0))
        distribution_averages = torch.stack(outputs_list,dim = 1)
        outputs = distribution_averages
        return outputs, {'logits':intermediary_logits_list, 'averages':distribution_averages}

class FitSigmoidOutput(nn.Module):
    def __init__(self, bins):
        super(FitSigmoidOutput, self).__init__()
        assert(configs['network_output_kind']=='linear')
        self.bins = bins.squeeze(0).cpu().numpy()
    
    def sigmoid(self,x,x0,k):
        return 1/(1+np.exp(k*(x-x0)))
    
    def set_bins(self, bins):
        self.bins = bins.squeeze(0).cpu().numpy()
            
    def forward(self, input):
        output = []
        for i in range(input.size(0)):
            #print(input[i].detach().cpu().numpy().shape)
            #print(self.bins.shape)
            try:
                popt, pcov = curve_fit(self.sigmoid, self.bins, input[i].detach().cpu().numpy(), p0=[0.7, 5])
                output.append(popt[0])
            except RuntimeError:
                output.append(0.0)
        #print(torch.tensor(output).size())
        return torch.tensor(output).cuda().float().view(-1,1)
        
class FitStepOutput(nn.Module):
    def __init__(self, bins):
        super(FitStepOutput, self).__init__()
        
        assert(configs['network_output_kind']=='linear')
        self.bins = bins.squeeze(0)
    
    def set_bins(self, bins):
        self.bins = bins.squeeze(0)
            
    def forward(self, input):
        scores = torch.zeros([input.size(0), len(self.bins)+1]).cuda()
        scores[:,0] = torch.zeros([input.size(0)]).cuda() #torch.sum((input)**2, dim = 1)
        for i in range(len(self.bins)):
            scores[:,i+1] = scores[:,i] + 1 - 2*input[:,i]
        indices = torch.argmin(scores, dim = 1)
        bins_to_select = torch.cat([self.bins[:1], self.bins, self.bins[-1:]])
        chosen_values = (bins_to_select[indices] + bins_to_select[indices+1])/2
        return chosen_values.unsqueeze(1)

class BinaryClassifiersOutput(nn.Module):
    def __init__(self, in_features, bins):
        super(BinaryClassifiersOutput, self).__init__()
        self.bins = bins
        intermediary_probabilities_list = []
        outputs_list = []
        for index, output_column_name in enumerate(configs['get_labels_columns']):
            output_size_linear_layer= configs['n_binary_classifiers_when_percentile'] if configs['binary_classifiers_percentile_spacing'] else bins[index].size()[1]
            one_output = nn.Sequential()
            one_output.add_module('linear_to_logits', nn.Linear(in_features, output_size_linear_layer))

            intermediary_probabilities_list.append(one_output)
            one_output = nn.Sequential()
            
            if configs['post_binary_classifiers_fit_or_linear']=='fit':
                one_output.add_module('logits_to_probs', nn.Sigmoid())
                if configs['binary_classifiers_fit_type']=='sigmoid':
                    one_output.add_module('fit_sigmoid', FitSigmoidOutput(self.bins[index]))
                elif configs['binary_classifiers_fit_type']=='step':
                    one_output.add_module('fit_sigmoid', FitStepOutput(self.bins[index]))
            elif configs['post_binary_classifiers_fit_or_linear']=='linear':
                for i in range(configs['binary_classifiers_n_post_layers']-1):
                    one_output.add_module('binary_classifier_inter_layer_' + str(i), nn.Linear(output_size_linear_layer, output_size_linear_layer))
                    one_output.add_module('binary_classifier_inter_layer_relu_' + str(i), nn.ReLU())
                one_output.add_module('probs_to_mean', nn.Linear(output_size_linear_layer, 1))
            
            outputs_list.append(one_output)
        self.intermediary_probabilities_module_list = nn.ModuleList(intermediary_probabilities_list)
        self.outputs_module_list = nn.ModuleList(outputs_list)
    
    def set_bins(self, bins, index):
        self.bins[index] = bins
        self.outputs_module_list[index].fit_sigmoid.set_bins(bins)
    
    def forward(self, input):
        intermediary_logits_list = []
        outputs_list = []
        size0 = input.size()[0]
        for index, output_column_name in enumerate(configs['get_labels_columns']):
            #TODO: if output kind is sigmoid, or maybe if output is copd, skip next two lines
            intermediary_logits_one_output = self.intermediary_probabilities_module_list[index](input)
            intermediary_logits_list.append(intermediary_logits_one_output)
            one_output = self.outputs_module_list[index](intermediary_logits_one_output)
            outputs_list.append(one_output.squeeze(1))
        distribution_averages = torch.stack(outputs_list,dim = 1)
        outputs = distribution_averages
        return outputs, {'logits':intermediary_logits_list, 'averages':distribution_averages}

class LineClassifiersOutput(nn.Module):
    def __init__(self, in_features, bins):
        super(LineClassifiersOutput, self).__init__()
        self.bins = bins
        intermediary_probabilities_list = []
        outputs_list = []
        for index, output_column_name in enumerate(configs['get_labels_columns']):
            output_size_linear_layer= configs['n_binary_classifiers_when_percentile'] if configs['binary_classifiers_percentile_spacing'] else bins[index].size()[1]
            one_output = nn.Sequential()
            one_output.add_module('linear_to_logits', nn.Linear(in_features, output_size_linear_layer))

            intermediary_probabilities_list.append(one_output)
            one_output = nn.Sequential()
            one_output.add_module('probs_to_mean', nn.Linear(output_size_linear_layer, 2))
            #one_output.add_module('logits_to_probs', nn.Softplus())
            
            outputs_list.append(one_output)
        self.intermediary_probabilities_module_list = nn.ModuleList(intermediary_probabilities_list)
        self.outputs_module_list = nn.ModuleList(outputs_list)

    def forward(self, input):
        intermediary_logits_list = []
        outputs_list = []
        size0 = input.size()[0]
        for index, output_column_name in enumerate(configs['get_labels_columns']):
            #TODO: if output kind is sigmoid, or maybe if output is copd, skip next two lines
            intermediary_logits_one_output = self.intermediary_probabilities_module_list[index](input)
            one_output = self.outputs_module_list[index](intermediary_logits_one_output)
            #intermediary_logits_list.append(-nn.Softplus()(one_output[:,0]).unsqueeze(1)*self.bins[index] + one_output[:,1].unsqueeze(1))
            intermediary_logits_list.append(-nn.Softplus()(one_output[:,0]).unsqueeze(1)*self.bins[index] + one_output[:,1].unsqueeze(1).detach()*nn.Softplus()(one_output[:,0]).unsqueeze(1))
            #print(intermediary_logits_list[index])
            #outputs_list.append(one_output[:,1]/nn.Softplus()(one_output[:,0]))
            outputs_list.append(one_output[:,1])
        distribution_averages = torch.stack(outputs_list,dim = 1)
        outputs = distribution_averages
        return outputs, {'logits':intermediary_logits_list, 'averages':distribution_averages}
        
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
            #linear_input_features = configs['dsnm_pca_size']
            linear_input_features = self.current_n_channels
        else:
            linear_input_features = self.current_n_channels
        self.dsnm = nn.Sequential(nn.Linear(linear_input_features, self.n_groups*(self.n_groups-1)),MultiplyInGroups(self.n_groups))
        self.current_n_channels = self.n_groups
        if configs['use_extra_inputs']:
            self.current_n_channels = self.current_n_channels+15
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

        w,b, self.kmeans = get_unsupervised_weights(pcaed_images, self.n_groups, 'dsnmkmeansout', self.pca)

        self.dsnm.apply(lambda tensor: init_k_means(tensor, w, b))

        self.initialized = True

    def forward(self, x, extra_fc_input = None):
        if not self.initialized:
            if self.acumulated_x is None:
                self.acumulated_x = x.data.cpu().numpy()
            else:
                self.acumulated_x = np.concatenate((self.acumulated_x, x.data.cpu().numpy()), axis=0)
            return (x*0)[:,0:self.current_n_channels], None
        if configs['use_pca_dsnm']:
            x = self.pca_forward(x)
        #print(Counter(self.kmeans.predict(x.data.cpu().numpy())))
        #print(Counter(self.kmeans.predict(self.pca_forward(x).data.cpu().numpy())))
        x = self.dsnm(x)
        #print(torch.max(x, dim = 1)[1])
        if extra_fc_input is not None:
            x = torch.cat((extra_fc_input,x),1)
        return x, {'ws':None,'vs':None}

class ModelSpatialToFlatPart(nn.Module):
    def __init__(self, num_ftrs):
        super(ModelSpatialToFlatPart, self).__init__()
        
        _, self.n_cnns = get_qt_inputs()
        spatial_parts = []
        for i in range(self.n_cnns):
            this_spatial_part, self.current_n_channels = get_spatial_part_fc(num_ftrs, i)
            spatial_parts.append(this_spatial_part)
        self.spatial_part = nn.ModuleList(spatial_parts)
        self.spatial_part.apply(weights_init)
        
        if configs['use_delayed_lateral_pooling']:
            self.spatial_part2 = nn.Sequential()
            self.spatial_part2.add_module("conv1avgpool",nn.Conv2d(kernel_size = (3,1), in_channels = 1024, out_channels =  256, padding=(1,0),bias=True).cuda())
            self.spatial_part2.add_module("reluavgpool",nn.ReLU(inplace = True).cuda())
            self.spatial_part2.add_module("conv2avgpool",nn.Conv2d(kernel_size = (3,1), in_channels = 256, out_channels =  256, padding=(1,0),bias=True).cuda())
            self.spatial_part2.add_module("avgpool2",nn.AvgPool2d(kernel_size = (configs['avg_pool_kernel_size'],1)).cuda())
            self.current_n_channels = 256
    
    def forward(self, spatial_outputs):
        all_outputs = []
        for i, spatial_output in enumerate(spatial_outputs):
            if configs['tie_conv11_same_weights']:
                index = 0
            else:
                index = i
            b = self.spatial_part[index](spatial_output.contiguous())
            if not configs['use_delayed_lateral_pooling']:
                b = b.view(b.size(0), -1)
            all_outputs.append(b)
        x = torch.cat(all_outputs, 1)
        if configs['use_delayed_lateral_pooling']:
            x = self.spatial_part2(x)
            x = x.view(x.size(0), -1)
        return x,spatial_outputs

class LogQuotientOutput(nn.Module):
    def __init__(self, current_n_channels):
        super(LogQuotientOutput, self).__init__()
        self.softplus = torch.nn.Softplus()
        self.linear = nn.Linear(current_n_channels, 2*len(labels_columns))
    
    def forward(self, input):
        x = self.linear(input)
        x = x.view([-1, 2,len(labels_columns)])
        x2 = torch.log(self.softplus(x[:,0,:])/self.softplus(x[:,1,:]))
        if self.training :
            grad_to_improve, = torch.autograd.grad(x2, x,
                               grad_outputs=x2.data.new(x2.shape).fill_(1),
                               create_graph=True)
        else:
            grad_to_improve = x.data.new(x.shape).fill_(0)
        return x2, {'grad_to_improve':grad_to_improve}

class ContinuousBinaryClassifiersOutput(nn.Module):
    def __init__(self, current_n_channels):
        super(ContinuousBinaryClassifiersOutput, self).__init__()
        self.softplus = torch.nn.Softplus()
        self.linear = nn.Linear(current_n_channels, 2*len(labels_columns))
    
    def forward(self, input):
        x = self.linear(input)
        x = x.view([-1, 2,len(labels_columns)])
        x2 = torch.log(self.softplus(x[:,0,:])/self.softplus(x[:,1,:]))
        if self.training :
            grad_to_improve, = torch.autograd.grad(x2, x,
                               grad_outputs=x2.data.new(x2.shape).fill_(1),
                               create_graph=True)
        else:
            grad_to_improve = x.data.new(x.shape).fill_(0)
        return x2, {'grad_to_improve':grad_to_improve}
        
class CommonLinearOutput(nn.Module):
    def __init__(self, current_n_channels):
        super(CommonLinearOutput, self).__init__()
        self.linear = nn.Linear(current_n_channels , len(labels_columns)).cuda()
    
    def forward(self, x):
        return self.linear(x), {}

class ModelLastLinearLayer(nn.Module):
    def __init__(self, current_n_channels):
        super(ModelLastLinearLayer, self).__init__()
        self.current_n_channels = current_n_channels
        self.final_linear_layer = torch.nn.Sequential()

        if configs['dropout_batch_normalization_last_layer']:
            if configs['use_batchnormalization_hidden_layers']:
                self.final_linear_layer.add_module("bn_out",torch.nn.BatchNorm1d(self.current_n_channels).cuda())
            self.final_linear_layer.add_module("drop_out",nn.Dropout(p=configs['use_dropout_hidden_layers']).cuda())

        if configs['use_mean_var_loss'] or configs['use_binary_classifiers']:
            self.bins_list = []
            for index, col_name in enumerate(configs['get_labels_columns']):
                if configs['use_sigmoid_safety_constants']:
                    min_range = configs['pre_transform_labels'].ranges_labels[col_name][0]*configs['sigmoid_safety_constant'][col_name][0]
                    max_range = configs['pre_transform_labels'].ranges_labels[col_name][1]*configs['sigmoid_safety_constant'][col_name][1]
                else:
                    min_range = configs['pre_transform_labels'].ranges_labels[col_name][0]
                    max_range = configs['pre_transform_labels'].ranges_labels[col_name][1]
                spacing = configs['meanvarloss_discretization_spacing']
                self.bins_list.append(torch.autograd.Variable(torch.arange(min_range, max_range+spacing, spacing).unsqueeze(0).cuda(non_blocking=True, device = 0), requires_grad=False) )
            if configs['use_binary_classifiers']:
                linear_out_model = BinaryClassifiersOutput(self.current_n_channels, self.bins_list).cuda()
            else:
                linear_out_model = MeanVarLossOutput(self.current_n_channels, self.bins_list).cuda()
        elif configs['use_log_quotient_output']:
            linear_out_model = LogQuotientOutput(self.current_n_channels).cuda()
        else:
            linear_out_model = CommonLinearOutput(self.current_n_channels).cuda()
        
        self.final_linear_layer.add_module("linear_out", linear_out_model)
        self.final_linear_layer.apply(weights_init)
    
    def set_bins(self, bins, index):
        self.bins_list[index] = bins
        if configs['use_mean_var_loss'] or configs['use_binary_classifiers']:
            self.final_linear_layer.linear_out.set_bins(bins,index)
    
    def forward(self, input):
        x, extra_outs = self.final_linear_layer(input)
        output_kind_each_output = [ configs['get_individual_output_kind'][name] for name in configs['get_labels_columns']]
        dic_output_kinds = {'linear':nn.Sequential(),'softplus':nn.Sequential(nn.Softplus().cuda()), 'sigmoid':nn.Sequential(nn.Sigmoid().cuda())}
        #add exception when output_kind_each_output cotains element not in dic_output_kinds.keys()
        unrecognized_kinds_of_outputs = list(set(output_kind_each_output).difference( dic_output_kinds.keys()) )
        if len(unrecognized_kinds_of_outputs)>0:
            raise_with_traceback(ValueError('There are output kinds in configs["individual_output_kind"] or configs["network_output_kind"] that are not one of: linear, sigmoid and softplus: ' + str(unrecognized_kinds_of_outputs)))
        all_masked_outputs = []
        for output_kind in list(dic_output_kinds.keys()):
            if utils.compare_versions(torch.__version__, '0.4.0')>=0:
                torch.set_grad_enabled(False)
            mask = torch.autograd.Variable(torch.FloatTensor(np.repeat(np.expand_dims([(1.0 if output_kind_each_output[k] == output_kind else 0.0) for k in range(len(output_kind_each_output))], axis = 0), dic_output_kinds[output_kind](x).size()[0], axis=0)).cuda(), volatile = False)
            if utils.compare_versions(torch.__version__, '0.4.0')>=0:
                torch.set_grad_enabled(True)
            all_masked_outputs.append((dic_output_kinds[output_kind](x)*mask).unsqueeze(2))
        x = torch.cat(all_masked_outputs, 2)
        x = torch.sum(x, 2)
        if len(x.size())>2:
            x = x.squeeze(2)
        return x, extra_outs

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

        extra_input_size = 15 #TEMP replace 15 with 42 if the extrainput is the segmentation features
        if configs['extra_fc_layers_for_extra_input']:
            self.extra_fc_layers_for_extra_input = nn.Sequential(
                        nn.BatchNorm1d(extra_input_size),
                        nn.Linear(extra_input_size, configs['extra_fc_layers_for_extra_input_output_size']),
                        nn.ReLU(),
                        nn.Linear(configs['extra_fc_layers_for_extra_input_output_size'], configs['extra_fc_layers_for_extra_input_output_size']),
                        nn.ReLU())
            extra_input_size = configs['extra_fc_layers_for_extra_input_output_size']

        current_model = self.fc_before_extra_inputs
        for layer_i in range(configs['n_hidden_layers']):
            if (configs['layer_to_insert_extra_inputs']== layer_i) and configs['use_extra_inputs']:
                if configs['normalize_extra_inputs_and_rest_with_bn']:
                    self.bn_rest = nn.BatchNorm1d(self.current_n_channels)
                    self.bn_extra_input = nn.BatchNorm1d(extra_input_size)
                self.current_n_channels = self.current_n_channels+extra_input_size
                #self.current_n_channels = extra_input_size #TEMP remove; for when testing using only the extra inputs
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
            if configs['normalize_extra_inputs_and_rest_with_bn']:
                self.bn_rest = nn.BatchNorm1d(self.current_n_channels)
                self.bn_extra_input = nn.BatchNorm1d(extra_input_size)
            self.current_n_channels = self.current_n_channels+extra_input_size
            #self.current_n_channels = extra_input_size #TEMP remove
            current_model = self.fc_after_extra_inputs

        if configs['use_extra_inputs'] and configs['layer_to_insert_extra_inputs']>(configs['n_hidden_layers']+1):
            raise_with_traceback(ValueError("configs['layer_to_insert_extra_inputs'] ("+configs['layer_to_insert_extra_inputs']+") is set to bigger than configs['n_hidden_layers']+1 ("+configs['n_hidden_layers']+1 +")"))
        self.fc_after_extra_inputs.apply(weights_init)
        self.fc_before_extra_inputs.apply(weights_init)


    def forward(self, input, extra_fc_input = None):
        x = self.fc_before_extra_inputs(input)
        if extra_fc_input is not None:
            if configs['extra_fc_layers_for_extra_input']:
                extra_fc_input = self.extra_fc_layers_for_extra_input(extra_fc_input)
            if configs['normalize_extra_inputs_and_rest_with_bn']:
                x = self.bn_rest(x)
                extra_fc_input = self.bn_extra_input(extra_fc_input)
            x = torch.cat((extra_fc_input,x),1) #extra_fc_input #torch.cat((extra_fc_input,x),1) #TEMP
        x = self.fc_after_extra_inputs(x)
        return x, {'ws':None,'vs':None}

class SoftmaxWithIdentityGradient(torch.autograd.Function):
    def __init__(self):
        super(SoftmaxWithIdentityGradient, self).__init__()

    def forward(self, input):
        #return nn.functional.softmax(torch.autograd.Variable(input), dim = 1).data
        if utils.compare_versions(torch.__version__, '0.4.0')>=0:
            nonlinearity = nn.Softmax(dim = 1)
        else:
            nonlinearity = nn.Softmax()
        return nonlinearity(torch.autograd.Variable(input)).data

    def backward(self, grad_output):
        grad_input = grad_output.clone()/grad_output.size(1)
        return grad_input

def swig(input):
    return SoftmaxWithIdentityGradient()(input)

class ModelInternalClassSelection(nn.Module):
    def __init__(self, current_n_channels):
        super(ModelInternalClassSelection, self).__init__()
        self.current_n_channels = current_n_channels
        activation_function_dict = {'relu': nn.ReLU().cuda(), 'tanh':nn.Tanh().cuda(),
                            'sigmoid':nn.Sigmoid().cuda(), 'softplus':nn.Softplus().cuda()
        }

        activation_function = activation_function_dict[configs['fc_activation']]

        self.fc_1 = torch.nn.Sequential()

        if configs['use_batchnormalization_hidden_layers']:
            self.fc_1.add_module("bn_fc_1",torch.nn.BatchNorm1d(self.current_n_channels).cuda())
        self.fc_1.add_module("drop_fc_1",nn.Dropout(p=configs['use_dropout_hidden_layers']).cuda())
        # On DGX, this next line, in the first iteration of the loop, is the one taking a long time (about 250s) in the .cuda() part.
        # probably because of some incompatibility between Cuda 9.0 and pytorch 0.1.12
        self.fc_1.add_module("linear_fc_1",nn.Linear(self.current_n_channels, configs['channels_hidden_layers'] ).cuda())
        self.fc_1.add_module("nonlinearity_fc_1",activation_function)
        self.current_n_channels = configs['channels_hidden_layers']

        self.fc_11 = torch.nn.Sequential()

        if configs['use_batchnormalization_hidden_layers']:
            self.fc_11.add_module("bn_fc11",torch.nn.BatchNorm1d(self.current_n_channels).cuda())
        self.fc_11.add_module("drop_fc11",nn.Dropout(p=configs['use_dropout_hidden_layers']).cuda())
        # On DGX, this next line, in the first iteration of the loop, is the one taking a long time (about 250s) in the .cuda() part.
        # probably because of some incompatibility between Cuda 9.0 and pytorch 0.1.12
        self.fc_11.add_module("linear_fc11",nn.Linear(self.current_n_channels, 512 ).cuda())
        self.fc_11.add_module("nonlinearity_fc_11",activation_function)
        self.fc_11.add_module("linear_fc11_2",nn.Linear(512, configs['classes_hidden_layers'] ).cuda())
        #self.fc_11.add_module("nonlinearity_fc11",nn.Softplus())
        if utils.compare_versions(torch.__version__, '0.4.0')>=0:
            self.fc_11.add_module("nonlinearity_fc11",nn.Softmax(dim = 1))
        else:
            self.fc_11.add_module("nonlinearity_fc11",nn.Softmax())
        #self.fc_11.add_module("nonlinearity_fc11",nn.Sigmoid())

        fcs = []
        for i in range(configs['classes_hidden_layers']):
            fc_to_add = torch.nn.Sequential()

            if configs['use_batchnormalization_hidden_layers']:
                fc_to_add.add_module("bn_fc_12"+str(i),torch.nn.BatchNorm1d(self.current_n_channels).cuda())
            #fc_to_add.add_module("drop_fc_12"+str(i),nn.Dropout(p=configs['use_dropout_hidden_layers']).cuda())
            fc_to_add.add_module("linear_fc_12"+str(i),nn.Linear(self.current_n_channels, 512 ).cuda())
            fc_to_add.add_module("nonlinearity_fc_12"+str(i),activation_function)
            fc_to_add.add_module("linear_fc_12_2"+str(i),nn.Linear(512, 512 ).cuda())
            fcs.append(fc_to_add)

        self.fc_12 = nn.ModuleList(fcs)

        self.fc_1.apply(weights_init)
        self.fc_12.apply(weights_init)
        self.fc_11.apply(weights_init)

        if configs['use_extra_inputs']:
                self.current_n_channels = self.current_n_channels+15

        #TODO: do random fixed dropout of each input feature of the non-softmax branch
        #torch.randint_like()

        #TODO: orthogonal loss


    def forward(self, input, extra_fc_input = None, epoch=0):

        x1 = self.fc_1(input)

        '''
        if epoch < 10:
            ws = self.fc_11(x.detach())
        else:
            ws = self.fc_11(x)
        '''

        ws = self.fc_11(x1)
        #ws = swig(ws)
        print(torch.max(ws, dim = 1)[1])
        vs = []
        for i in range(configs['classes_hidden_layers']):
            v = self.fc_12[i](x1)
            vs.append(v)
            v = ws[:,i].unsqueeze(1).expand(x1.size(0), v.size(1))*v
            if i ==0:
                x = v
            else:
                x = x + v
        vs = torch.stack(vs,dim = 2)
        if extra_fc_input is not None:
            x = torch.cat((extra_fc_input,x),1)
        return x, {'ws':ws,'vs':vs}

class NonDataParallel(nn.Module):
    def __init__(self, model):
        super(NonDataParallel, self).__init__()
        self.module = model

    def forward(self, input1, input2, extra_inputs, epoch):

        return self.module(input1, input2, extra_inputs, epoch)

def get_model(num_ftrs):
    outmodel = ModelMoreInputs(num_ftrs)
    if configs['use_more_one_gpu']:
        outmodel = torch.nn.DataParallel(outmodel).cuda()
    else:
        outmodel = NonDataParallel(outmodel).cuda()
    if configs['load_model']:
        state_to_load = torch.load(configs['local_directory'] + '/models/'+configs['prefix_model_to_load']+'model' + configs['model_to_load'] + '_0')
        #state_to_load['module.final_layers.final_linear.final_linear_layer.linear_out.linear.weight'] = torch.cat([state_to_load['module.final_layers.final_linear.final_linear_layer.linear_out.linear.weight'], torch.zeros_like(state_to_load['module.final_layers.final_linear.final_linear_layer.linear_out.linear.weight'])[0:1, :]], dim = 0)
        #state_to_load['module.final_layers.final_linear.final_linear_layer.linear_out.linear.bias'] = torch.cat([state_to_load['module.final_layers.final_linear.final_linear_layer.linear_out.linear.bias'], torch.zeros_like(state_to_load['module.final_layers.final_linear.final_linear_layer.linear_out.linear.bias'])[0:1]], dim = 0)
        outmodel.load_state_dict(state_to_load)
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

class BasicBlockWithDropout(torchvision.models.resnet.BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockWithDropout, self).__init__( inplanes, planes, stride, downsample)

        self.dropout = nn.Dropout2d(p=configs['densenet_dropout'])

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def getMyResnet18(pretrained=False, **kwargs):
    model = torchvision.models.ResNet(BasicBlockWithDropout, [2,2,2,2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
    return model

#from https://gist.github.com/guillefix/23bff068bdc457649b81027942873ce5
class Conv2dLocal(nn.Module):

    def __init__(self, in_height, in_width, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2dLocal, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.in_height = in_height
        self.in_width = in_width
        self.out_height = int(math.floor(
            (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        self.out_width = int(math.floor(
            (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        self.weight = torch.nn.parameter.Parameter(torch.Tensor(
            self.out_height, self.out_width,
            out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.Tensor(
                out_channels, self.out_height, self.out_width))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, input):
        return conv2d_local(
            input, self.weight, self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation)

#from https://gist.github.com/guillefix/23bff068bdc457649b81027942873ce5
def conv2d_local(input, weight, bias=None, padding=0, stride=1, dilation=1):
    if input.dim() != 4:
        raise NotImplementedError("Input Error: Only 4D input Tensors supported (got {}D)".format(input.dim()))
    if weight.dim() != 6:
        # outH x outW x outC x inC x kH x kW
        raise NotImplementedError("Input Error: Only 6D weight Tensors supported (got {}D)".format(weight.dim()))

    outH, outW, outC, inC, kH, kW = weight.size()
    kernel_size = (kH, kW)

    # N x [inC * kH * kW] x [outH * outW]
    cols = F.unfold(input, kernel_size, dilation=dilation, padding=padding, stride=stride)
    cols = cols.view(cols.size(0), cols.size(1), cols.size(2), 1).permute(0, 2, 3, 1)

    out = torch.matmul(cols, weight.view(outH * outW, outC, inC * kH * kW).permute(0, 2, 1))
    out = out.view(cols.size(0), outH, outW, outC).permute(0, 3, 1, 2)

    if bias is not None:
        out = out + bias.expand_as(out)
    return out

class CheXNet(nn.Module):
    def __init__(self, out_size, num_layers = 121, architecture = 'densenet'):
        super(CheXNet, self).__init__()
        self.architecture = architecture
        if configs['pretrain_kind'] == 'imagenet':
            model_parameters = {'pretrained':True}
        else:
            model_parameters = {'pretrained':False}
        if architecture=='densenet':
            model_parameters['drop_rate'] = configs['densenet_dropout']
            num_layers_to_model = {121:torchvision.models.densenet121, 169:torchvision.models.densenet169, 201:torchvision.models.densenet201, 161:torchvision.models.densenet161}
        elif architecture=='resnet':
            num_layers_to_model = {18:torchvision.models.resnet18, 34:torchvision.models.resnet34, 50:torchvision.models.resnet50, 101:torchvision.models.resnet101, 152:torchvision.models.resnet152}
            #num_layers_to_model = {18:getMyResnet18, 34:torchvision.models.resnet34, 50:torchvision.models.resnet50, 101:torchvision.models.resnet101, 152:torchvision.models.resnet152}
        if architecture=='vgg':
            if configs['vgg_batch_norm']:
                num_layers_to_model = {11:torchvision.models.vgg11_bn, 13:torchvision.models.vgg13_bn, 16:torchvision.models.vgg16_bn, 19:torchvision.models.vgg19_bn}
            else:
                num_layers_to_model = {11:torchvision.models.vgg11, 13:torchvision.models.vgg13, 16:torchvision.models.vgg16, 19:torchvision.models.vgg19}
        if architecture=='alexnet':
            num_layers_to_model = {5:torchvision.models.alexnet}
        if architecture=='inception':
            num_layers_to_model = {48:torchvision.models.inception_v3}
        if architecture=='squeezenet':
            if configs['squeezenet_version_11']:
                num_layers_to_model = {18:torchvision.models.squeezenet1_1}
            else:
                num_layers_to_model = {18:torchvision.models.squeezenet1_0}

        self.model = num_layers_to_model[num_layers](**model_parameters)
        if self.architecture=='vgg':
            self.num_ftrs = self.get_classifier()[0].in_features/configs['avg_pool_kernel_size']/configs['avg_pool_kernel_size']
        elif self.architecture=='alexnet':
            self.num_ftrs = self.get_classifier()[1].in_features/configs['avg_pool_kernel_size']/configs['avg_pool_kernel_size']
        elif self.architecture=='squeezenet':
            self.num_ftrs = 512
        else:
            self.num_ftrs = self.get_classifier().in_features

        if configs['use_local_conv']:
            self.num_ftrs = configs['n_channels_local_convolution']
        new_last_layer = nn.Sequential(
            nn.Linear(self.num_ftrs, out_size)
            #, nn.Sigmoid()
        )

        self._set_classifier(new_last_layer)

        self.modified_end_avg_pool = False

    def get_classifier(self):
        if self.architecture=='densenet' or self.architecture == 'vgg' or self.architecture=='alexnet' or self.architecture=='squeezenet' :
            return self.model.classifier
        elif self.architecture=='resnet' or self.architecture == 'inception':
            return self.model.fc

    def _set_classifier(self, new_last_layer):
        if self.architecture=='densenet' or self.architecture == 'vgg' or self.architecture=='alexnet' or self.architecture=='squeezenet':
            self.model.classifier = new_last_layer
        elif self.architecture=='resnet' or self.architecture == 'inception':
            self.model.fc = new_last_layer

    def set_classifier_containing_avg_pool_part(self, classifier):
        self._set_classifier(classifier)
        #self.model.classifier = classifier
        self.modified_end_avg_pool = True
        #if self.architecture=='resnet':
        #    self.model.avgpool = nn.Sequential()

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
            if self.architecture=='inception':
                if self.model.transform_input:
                    x[:, 0] = -(-x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5) #TEMP: corrrection because input is inversed
                    x[:, 1] = -(-x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5) #TEMP: corrrection because input is inversed
                    x[:, 2] = -(-x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5) #TEMP: corrrection because input is inversed
                x = self.model.Conv2d_1a_3x3(x)
                x = self.model.Conv2d_2a_3x3(x)
                x = self.model.Conv2d_2b_3x3(x)
                x = F.max_pool2d(x, kernel_size=3, stride=2)
                x = self.model.Conv2d_3b_1x1(x)
                x = self.model.Conv2d_4a_3x3(x)
                x = F.max_pool2d(x, kernel_size=3, stride=2)
                x = self.model.Mixed_5b(x)
                x = self.model.Mixed_5c(x)
                x = self.model.Mixed_5d(x)
                x = self.model.Mixed_6a(x)
                x = self.model.Mixed_6b(x)
                x = self.model.Mixed_6c(x)
                x = self.model.Mixed_6d(x)
                x = self.model.Mixed_6e(x)
                x = self.model.Mixed_7a(x)
                x = self.model.Mixed_7b(x)
                x = self.model.Mixed_7c(x)
                #print(x.size())
            elif self.architecture=='densenet' or self.architecture=='vgg' or self.architecture=='alexnet' or self.architecture=='squeezenet':
                x = self.model.features(x)
                #x = F.relu(x, inplace=True) # TODO: should I always have this relu on?

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
            x = F.avg_pool2d(x, kernel_size=configs['avg_pool_kernel_size'], stride=1).view(x.size(0), -1)
        x = self.model.classifier(x)
        return x
'''

def get_unsupervised_weights(inputs, nClusters, filename, pca):
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
    N_CHANNELS = inputs.shape[1] #50176
    kmeans = KMeans(n_clusters=nClusters, init='k-means++').fit(inputs)
    print(Counter(kmeans.labels_))
    cc = np.matmul(kmeans.cluster_centers_,pca.components_)+pca.mean_
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

    return w, b, kmeans

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

'''
ACTIVATION = nn.ReLU

class Identity(nn.Module):

    def forward(self, x):
        return x

def conv2d_block(in_channels, out_channels, kernel=3, stride=1, padding=1, activation=ACTIVATION):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
    )

class C3DFCNout(nn.Module):
    def __init__(self, n_channels=1, init_filters=16, dimensions=2, batch_norm=False):
        super(C3DFCNout, self).__init__()
        self.cnn = nn.ModuleList([C3DFCN(n_channels, init_filters, dimensions, batch_norm)])
        self.stn = nn.Sequential()
        self.segmentation = nn.Sequential()
        self.final_layers = nn.Sequential()
        
    def forward(self,x, a,b,c):
        x = x[:,0:1,:,:]
        x = image_preprocessing.BatchUnNormalizeTensor([0.485], [0.229])(x)
        x = (x - 0.5)*2
        x = self.cnn[0](x)
        x = nn.Softplus()(x-50)*0.7
        x = torch.cat([x.unsqueeze(1),x.unsqueeze(1)], dim = 1)
        return x, {}

class C3DFCN(nn.Module):
    def __init__(self, n_channels=1, init_filters=16, dimensions=2, batch_norm=False):
        super(C3DFCN, self).__init__()
        nf = init_filters
        conv_block = conv2d_bn_block if batch_norm else conv2d_block
        max_pool = nn.MaxPool2d if int(dimensions) is 2 else nn.MaxPool3d
        self.encoder = nn.Sequential(
            conv_block(n_channels, nf),
            max_pool(2),
            conv_block(nf, 2*nf),
            max_pool(2),
            conv_block(2*nf, 4*nf),
            conv_block(4*nf, 4*nf),
            max_pool(2),
            conv_block(4*nf, 8*nf),
            conv_block(8*nf, 8*nf),
            max_pool(2),
            conv_block(8*nf, 16*nf),
            conv_block(16*nf, 16*nf),
            conv_block(16*nf, 16*nf),
        )
        self.classifier = nn.Sequential(
            conv_block(16*nf, 1, kernel=1, activation=Identity),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1).mean(1)
'''