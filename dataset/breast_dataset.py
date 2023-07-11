import os
import cv2
import pandas as pd
from skimage.exposure import equalize_adapthist
from torch.utils.data import DataLoader, Dataset, ConcatDataset

class BreastDataset(Dataset):

    '''
    INBreast dataset
    '''
    def __init__(self, csv_file: str, 
                 root_dir: str, 
                 transform=None):
        '''
            Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        '''
        self.img_dir = csv_file
        self.img_data = pd.read_csv(csv_file)
        self.root = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.img_data['Cancer'])

    def pre_img(self, img):
        img_clahe = equalize_adapthist(img, clip_limit=0.04, nbins=256)
        return img_clahe

    def __getitem__(self, idx):
        if self.img_data['Cancer'][idx] == 0:
            img_path = os.path.join(self.root, 'images', self.img_data['Path'][idx])
        else:
            img_path = os.path.join(self.root, 'images', self.img_data['Path'][idx])
#         print(img_path)
        img = cv2.imread(img_path, 1)
#         img = self.pre_img(img)
        if self.transform != None:
            image_trans = self.transform(image=img)['image']
        else:
            image_trans = img
        #return
        final_image = image_trans
        label = self.img_data['Cancer'][idx]
        return final_image, label