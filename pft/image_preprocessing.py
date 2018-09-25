from PIL import ImageMath
from random import *
import torch
import numpy as np
from PIL import Image
import numbers
import torchvision.transforms as transforms
from future.utils import raise_with_traceback

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
        c, w, h = img.shape
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
        c, w, h = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = (h - th)//2
        j = (w - tw)//2
        return i, j, th, tw

class CropSideCenterNumpy(RandomCropNumpy):

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        c, w, h = img.shape
        side = randint(0,1)
        j = (w)//4
        i = 0
        
        return i, j, h, w//2
      
class CropOneSideNumpy(RandomCropNumpy):

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        c, w, h = img.shape
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
        c, w, h = img.size()
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