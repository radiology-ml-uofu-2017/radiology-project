from PIL import ImageMath
from random import *
import torch
import numpy as np
from PIL import Image

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
            #return np.flip(tensor, 2)
            return tensor[:,:,::-1].copy()
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'

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
        input_var = torch.autograd.Variable(tensor.view(1,3,224,224), volatile=True)
        out = self.model(input_var)
        return [np.squeeze(np.transpose(out.data.cpu().numpy(), (0,1,2,3)),axis = 0)]

    def __repr__(self):
        return self.__class__.__name__ + '()'    

def preprocess_image(imagePath, transform):
    imageData = Image.open(imagePath)
    return transform(imageData)
        
def preprocess_images(all_images, transformations):
    a = all_images.apply(lambda row: preprocess_image(row['filepath'], transformations),axis=1)
    all_images['preprocessed'] = a
    return all_images