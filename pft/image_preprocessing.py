from PIL import ImageMath
from random import *
import torch
import numpy as np
from PIL import Image
import numbers
import torchvision.transforms as transforms
from future.utils import raise_with_traceback
import skimage
from scipy.ndimage.interpolation import rotate
from skimage.filters import rank
from skimage.morphology import disk
import torchvision
import math
import time
from configs import configs
if not configs['use_random_crops']:
    import cv2

class UnNormalizeNumpy(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        return_tensor = np.zeros_like(tensor)
        i = 0
        for t, m, s in zip(tensor, self.mean, self.std):
            return_tensor[i,...] = (t*s)+m
            i = i + 1
        return return_tensor

class BatchUnNormalizeTensor(object):
    def __init__(self, mean, std, inversed_input_correction =  (configs['unary_input_multiplier'] == -1)):
        self.mean = mean
        self.std = std
        self.inversed_input_correction = inversed_input_correction

    def __call__(self, tensor):
        
        mean = torch.FloatTensor(self.mean).to(tensor.device).view([1,-1,1,1])
        std = torch.FloatTensor(self.std).to(tensor.device).view([1,-1,1,1])
        if self.inversed_input_correction:
            to_return = (1-((-tensor*std)+mean)) #TEMP: corrrection because input is inversed
        else:
            to_return = ((tensor*std)+mean) #TEMP: corrrection because input is inversed
        if torch.max(to_return)>1.0000001:
            print(torch.max(to_return))
        if torch.min(to_return)<-0.0000001:
            print(torch.min(to_return))
        assert(torch.max(to_return)<=1.0000001 and torch.min(to_return)>=-0.0000001)
        return to_return

class BatchNormalizeTensorMinMax01(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        maxes, _ = torch.max(tensor.view([tensor.size(0), -1]), dim = 1)
        mins, _ = torch.min(tensor.view([tensor.size(0), -1]), dim = 1)
        if len(tensor.size()) <= 4:
            to_return = (tensor - mins.unsqueeze(1).unsqueeze(1).unsqueeze(1))/(maxes.unsqueeze(1).unsqueeze(1).unsqueeze(1) - mins.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        elif len(tensor.size()) == 5:
            to_return = (tensor - mins.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1))/(maxes.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1) - mins.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1))
        return to_return
        
class BatchNormalizeTensor(object):
    def __init__(self, mean, std, inversed_input_correction = False):
        self.mean = mean
        self.std = std
        self.inversed_input_correction = inversed_input_correction

    def __call__(self, tensor):
        mean = torch.FloatTensor(self.mean).cuda().view([1,3,1,1])
        std = torch.FloatTensor(self.std).cuda().view([1,3,1,1])
        if self.inversed_input_correction:
            to_return = (tensor+mean-1)/std
        else:
            to_return = (tensor-mean)/std
        return to_return

class NormalizeNumpy(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        return_tensor = np.zeros_like(tensor)
        i = 0
        for t, m, s in zip(tensor, self.mean, self.std):
            return_tensor[i,...] = (t-m)/s
            i = i + 1
        return return_tensor

class CropBiggestCenteredInscribedSquare(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        longer_side = min(tensor.size)
        horizontal_padding = (longer_side - tensor.size[0]) / 2
        vertical_padding = (longer_side - tensor.size[1]) / 2
        return tensor.crop(
            (
                -horizontal_padding,
                -vertical_padding,
                tensor.size[0] + horizontal_padding,
                tensor.size[1] + vertical_padding
            )
        )

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Convert16BitToFloat(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        tensor.mode = 'I'
        return ImageMath.eval('im/256', {'im':tensor}).convert('RGB')
        return tensor.point(lambda i:i*(1./256)).convert('RGB')

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomScaleAugmentationNumpy(object):
    def __init__(self, scale_range):
        self.scale_range = scale_range

    def __call__(self, image):
        chosen_scale = self.scale_range[0] + np.random.rand()*(self.scale_range[1]-self.scale_range[0])
        image = np.transpose(image, (1,2,0))
        image_2 = skimage.transform.rescale(image, (chosen_scale, chosen_scale), mode = 'edge', preserve_range =True)
        image_2 = crop_or_pad_to(image_2, image.shape)
        to_return = np.transpose(image_2.astype(float), (2,0,1))
        return to_return

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomRotationAugmentationNumpy(object):
    def __init__(self, rotation_range_degrees):
        self.rotation_range = rotation_range_degrees

    def __call__(self, image):
        image_min = np.min(image)
        image_max = np.max(image)
        image = ((image - image_min)/(image_max-image_min)).astype(np.float)
        chosen_rotation = self.rotation_range[0] + np.random.rand()*(self.rotation_range[1]-self.rotation_range[0])

        image_2 = np.transpose(skimage.transform.rotate(np.transpose(image, (1,2,0)), chosen_rotation, clip = False,  preserve_range=True, cval = 0.0), (2,0,1))

        image_2 = (image_2)*(image_max-image_min)+image_min
        to_return = image_2.astype(np.float)
        return to_return

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomGammaAugmentationNumpy(object):
    def __init__(self, gamma_range):
        self.gamma_range = gamma_range

    def __call__(self, image):
        A = self.gamma_range[0]
        B = 2*math.log10(1/A)
        chosen_gamma = A*10**(B*np.random.rand())
        #chosen_gamma = self.gamma_range[0] + np.random.rand()*(self.gamma_range[1]-self.gamma_range[0])
        #chosen_gamma = 1.0
        image_min = np.min(image)
        image_max = np.max(image)
        image = (image - image_min)/(image_max-image_min)
        image_2 = image**chosen_gamma# skimage.exposure.adjust_gamma(image, chosen_gamma)
        image_2 = image_2*(image_max-image_min)+image_min
        to_return = image_2.astype(float)
        return to_return 

    def __repr__(self):
        return self.__class__.__name__ + '()'

class castTensor(object):

    def __call__(self, image):
        return torch.FloatTensor(image)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class HistogramEqualization(object):
    def __init__(self, normalization_mean, normalization_std, local = False, nbins = 256 ):
        self.local = local
        self.nbins = nbins
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std

    def __call__(self, img):
        for channel in range(img.shape[0]):  # equalizing each channel
            image_min = np.min(img[channel, :, :])
            image_max = np.max(img[channel, :, :])
            
            image_min = -(0-self.normalization_mean[channel])/self.normalization_std[channel]
            image_max = -(1-self.normalization_mean[channel])/self.normalization_std[channel]
            img[channel, :, :] = (img[channel, :, :] - image_min)/(image_max-image_min)
            if not self.local:
                img[channel, :, :] = skimage.exposure.equalize_hist(img[channel, :, :], nbins = self.nbins)

            else:
                a = disk(15)
                img[channel, :, :] = (rank.equalize( img[channel, :, :], selem = a))/255.
            
            img[channel, :, :] = img[channel, :, :]*(image_max-image_min)+image_min
            
        image_2 = img

        return image_2

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ResizeNumpy():
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        if img.shape[0]==3:
            return np.concatenate((np.expand_dims(cv2.resize(img[0,...], dsize=self.size), axis = 0),
            np.expand_dims(cv2.resize(img[1,...], dsize=self.size), axis = 0),
            np.expand_dims(cv2.resize(img[2,...], dsize=self.size), axis = 0)), axis = 0)
        elif img.shape[0]==1:
            return np.expand_dims(cv2.resize(img[0,...], dsize=self.size), axis = 0)


def crop_or_pad_to(image, final_size):
    assert(len(image.shape)==len(final_size))
    for i in range(len(image.shape)):

        padding_list = [[0,0]]*len(image.shape)
        if (-image.shape[i] + final_size[i])%2 == 0:
            a = (-image.shape[i] + final_size[i])//2
            size_to_pad = [a, a]
        else:
            size_to_pad = [(-image.shape[i] + final_size[i]+1)//2 , (-image.shape[i] + final_size[i]-1)//2]
        padding_list[i] = size_to_pad
        padding_list = np.array(padding_list)
        if image.shape[i]<final_size[i]:
            image = skimage.util.pad(image, padding_list, mode = 'constant', constant_values  = 0.0)
        elif image.shape[i]>final_size[i]:
            image = skimage.util.crop(image,-padding_list,copy = True)
    return image

class RandomHorizontalFlipNumpy(object):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, tensor):
        if random() < self.p:
            return tensor[:,:,::-1].copy()
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToNumpy(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.numpy()

    def __repr__(self):
        return self.__class__.__name__ + '()'

def crop_center(img,sh, sw, th, tw):
    return img[:,sh:sh+th,sw:sw+tw]

def pad(array, padding):
    if isinstance(padding, numbers.Number):
        padding = (int(padding))
    if len(padding)==4: #left, top, right, bottom
        pass
    elif len(padding)==2:
        padding = padding*2
    elif len(padding)==1:
        padding = padding*4
    else:
        raise_with_traceback(ValueError('padding has an invalid value in function pad. expected list or tuple of length 1, 2 or 4 or a number. received: '+str(reference_shape)))

    result = np.zeros((array.shape[0], array.shape[1]+ padding[1]+ padding[3], array.shape[2]+ padding[0]+ padding[2]))
    result[:,padding[1]:(array.shape[1]+padding[1]), padding[0]:(array.shape[2]+padding[0])] = array
    return result

class RandomCropNumpy():

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        c, h , w = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = randint(0, h - th)
        j = randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = pad(img, self.padding)

        # pad the width if needed

        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        sh, sw, th, tw = self.get_params(img, self.size)

        return crop_center(img,sh, sw, th, tw).copy()

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class CenterCropNumpy(RandomCropNumpy):

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        c, h, w = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = (h - th)//2
        j = (w - tw)//2
        return i, j, th, tw

class CropSideCenterNumpy(RandomCropNumpy):

    @staticmethod
    def get_params(img, output_size):
        c, h, w = img.shape
        side = randint(0,1)
        j = (w)//4
        i = 0

        return i, j, h, w//2

class CropOneSideNumpy(RandomCropNumpy):

    @staticmethod
    def get_params(img, output_size):
        c, h , w = img.shape
        side = randint(0,1)
        if side == 1:
            j = 0
        else:
            j = (w)//2
        i = 0
        return i, j, h, w//2

'''
class RandomCropNumpy(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        c, h, w = img.size()
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = randint(0, h - th)
        j = randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        sh, sw, th, tw = self.get_params(img, self.size)
        return crop_center(img,sh, sw, th, tw)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)
'''

class RandomHorizontalFlipTensor(object):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, tensor):
        if random() < self.p:
            idx = [i for i in range(tensor.size[2]-1, -1, -1)]

            idx = torch.LongTensor(idx)
            inverted_tensor = tensor.index_select(2, idx)

            return inverted_tensor
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ChexnetEncode(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, tensor):
        if utils.compare_versions(torch.__version__, '0.4.0')>=0:
            torch.set_grad_enabled(False)
        input_var = torch.autograd.Variable(tensor.view(1,3,224,224), volatile=True)
        if utils.compare_versions(torch.__version__, '0.4.0')>=0:
            torch.set_grad_enabled(True)
        out = self.model(input_var)
        return [np.squeeze(np.transpose(out.data.cpu().numpy(), (0,1,2,3)),axis = 0)]

    def __repr__(self):
        return self.__class__.__name__ + '()'

def preprocess_image(imagePath, transform):
    imageData = Image.open(imagePath)
    '''
    try:
        imageData = Image.open(imagePath)
    except IOError:
        print('Not possible to open this image: ' + imagePath)
        return None
    '''
    return transform(imageData)

def preprocess_images(all_images, transformations):
    a = all_images.apply(lambda row: preprocess_image(row['filepath'], transformations),axis=1)
    all_images['preprocessed'] = a
    return all_images

def preprocess_images_and_save(all_images, transformations, h5f):
    for index, row in all_images.iterrows():
        h5f['dataset_1'][index,...] = preprocess_image(row['filepath'], transformations)
    return h5f
