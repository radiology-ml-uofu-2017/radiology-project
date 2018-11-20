from __future__ import print_function
import torch
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import Dataset
import ntpath

#-------------------------------------------------------------------------------- 

class DatasetGenerator(Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathDatasetFile, transform):
    
        self.listImage = pathDatasetFile
        self.listImageLabels = [[x] for x in range(len(pathDatasetFile))]
        self.transform = transform
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        

        imagePath = self.listImage[index]['filepath']

        imageData = Image.open(imagePath)#.convert('RGB')
        imageData.mode = 'I'
        imageData = imageData.point(lambda i:i*(1./256)).convert('RGB')

        #imageData.save("resized_images/convert_image"+str(index)+".png")
        
        longer_side = min(imageData.size)
        horizontal_padding = (longer_side - imageData.size[0]) / 2
        vertical_padding = (longer_side - imageData.size[1]) / 2
        imageData = imageData.crop(
            (
                -horizontal_padding,
                -vertical_padding,
                imageData.size[0] + horizontal_padding,
                imageData.size[1] + vertical_padding
            )
        )
    
        
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        if self.transform != None: imageData = self.transform(imageData)

        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImage)
    
 #-------------------------------------------------------------------------------- 
    
    
pathFileTrain = '/home/sci/ricbl/Documents/projects/xray/train.txt'
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
      splitted_ids = splitted_filepath[-1].replace('-','_').split('_')
      print(splitted_ids)
      thisimage['patientid'] = int(splitted_ids[2])
      thisimage['sessionid'] = int(splitted_ids[4])
      thisimage['scanid'] = int(splitted_ids[6])
      
      position = splitted_filepath[-3].upper()
      if 'LAT' in position:
          position = 'LAT'
      elif 'PA' in position:
          position = 'PA'
      elif 'AP' in position:
          position = 'AP'
          
      thisimage['position'] = position
      listImage.append(thisimage)
fileDescriptor.close()
[print("%s\t%s\t%s\t%s\n"%(item['patientid'],item['sessionid'],item['scanid'],item['position'])) for item in listImage]
1/0
transformSequence = transforms.Compose([ transforms.Resize(size=(224)), 
                                        transforms.ToTensor()])
datasetTrain = DatasetGenerator(pathDatasetFile=listImagePaths, transform=transformSequence)
dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=1, shuffle=False,  num_workers=1)
for i, (input, target) in enumerate (dataLoaderTrain):
    print(torch.min(input))
    print(torch.max(input))
    print(input.size())
    torchvision.utils.save_image(input, "resized_images/" + ntpath.basename(listImagePaths[int(target.numpy()[0][0])]), range = (0,1), padding = 0)    
    
