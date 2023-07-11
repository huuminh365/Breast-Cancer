import os
import cv2
import pandas as pd
from skimage.exposure import equalize_adapthist
from torch.utils.data import DataLoader, Dataset, ConcatDataset

class DDSMDataset(Dataset):
    def __init__(self, 
                 csv_file : str,
                 root_dir: str,
                 transform=None):
        self.root_dir = root_dir
        self.total_imgs = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.total_imgs)
    
    def pre_img(self, img):
        img_clahe = equalize_adapthist(img, clip_limit=0.04, nbins=256)
        return img_clahe
    
    def __getitem__(self, idx):
        img_path = '/kaggle/input' + '/' + self.total_imgs['File_path'][idx]
        image = cv2.imread(img_path, 1)
        label = self.total_imgs['cancer'][idx]
#         image = self.pre_img(image)
        if self.transform != None:
            image_trans = self.transform(image=image)['image']
        else:
            image_trans = image
        #return
        final_image = image_trans
#         print(image_trans.shape)
        return final_image, label