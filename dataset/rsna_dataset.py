import os
import cv2
import numpy as np
import pandas as pd
from albumentations import *
from skimage.exposure import equalize_adapthist
from torch.utils.data import DataLoader, Dataset, ConcatDataset

class RSNADataset(Dataset):
    
    '''
    RSNA Dataset
    '''
    def __init__(self,csv_file: str,
                 root_dir : str,
                 transform=None,
                 is_train = True):
        '''
            Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        '''
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.is_train = is_train
        lst_path = ['33581__357843412.png', '43952__1362091739.png', '58535__544454454.png']
        self.dataframe = self.dataframe[~self.dataframe['Path'].isin(lst_path)]
        self.dataframe.reset_index(drop=True, inplace=True)
        if is_train:
            self.transform = transform
        else:
            self.transform = Compose([Resize(height=224,width=224,always_apply=True),
                                      ToTensorV2()])
    
    def __len__(self):
        return len(self.dataframe)
    
    
    def get_cdf(self, img):
        m = int(np.max(img))
        hist = np.histogram(img, bins=m+1, range=(0, m+1))[0]
        hist = hist/img.size
        cdf = np.cumsum(hist)
        return cdf

    # numpy
    def histogram_equalization(self, img):
        m = int(np.max(img))
        hist = np.histogram(img, bins=m+1, range=(0, m+1))[0]
        # bước 1: tính pdf
        hist = hist/img.size
        # bước 2: tính cdf
        cdf = np.cumsum(hist)
        # bước 3: lập bảng thay thế
        s_k = (255 * cdf)
        # ảnh mới
        img_new = np.array([s_k[i] for i in img.ravel()]).reshape(img.shape)
        return img_new


    def HE(self, img):
        '''
            input: gray image
            dtype: np.array
        '''
        img_equ= self.histogram_equalization(img)
        return img_equ

    def pre_img(self, img):
#         clahe = cv2.createCLAHE(clipLimit = 20)
#         img_clahe = clahe.apply(img) + 30
        img_clahe = equalize_adapthist(img, clip_limit=0.04, nbins=256)
#         HE_img = self.HE(img_clahe)
#         print(HE_img)
        return img_clahe

    def __getitem__(self,index):

        if self.dataframe['Cancer'][index] == 0:
            img_path = os.path.join(self.root_dir, 'images', self.dataframe['Path'][index])
        else:
            img_path = os.path.join(self.root_dir, 'images', self.dataframe['Path'][index])
#         print(img_path)
        image = cv2.imread(img_path,1)
#         image = self.pre_img(image)
        
        #transform image
        # return a dict have key 'image'
        if self.transform != None:
            image_trans = self.transform(image=image)['image']
        else:
            image_trans = image
            
        #return
        final_image = image_trans
        if self.is_train:
            label = self.dataframe.iloc[index]['Cancer'] 
            return final_image,label
        else: 
            return final_image