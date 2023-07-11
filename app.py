import cv2
import torch
import gradio as gr
import skimage
from model import FullModel
from albumentations.pytorch import ToTensorV2
from albumentations import *
PATH = r"C:\Users\huumi\Desktop\Subjects\HK8\DOAN_THUCTAP\BreastCancer\July_convnext_labelsmoothing=0.1-benign-0-224_labelsmoothing_300rsna_b16.pt"


transform = Compose([Resize(height=224,width=224,always_apply=True),
                    ToTensorV2(),
                    ])

def load_model(PATH):
    model = FullModel()
    PATH = PATH
    model.load_state_dict(torch.load(PATH))
    return model

model = load_model(PATH)

def read_img(img):
    # print(path_img.shape)
    # img = cv2.imread(path_img, 1)
    img_trans = transform(image=img)['image']
    img_trans = img_trans.unsqueeze(0).float()
    return img_trans
    # print(img_trans.unsqueeze(0).shape)
    # print(model(img_trans.unsqueeze(0).float()))

def predict(img):
    result = "Malignant"
    images = read_img(img)
    print(images.shape)
    # images = images.reshape((-1, 3, 224, 224))
    # images = torch.from_numpy(images)#.permute(0, 2, 3, 1)
    outputs= model(images)
    print(outputs)
    _, predicted = torch.max(outputs, 1)
    if(predicted.item() == 0):
        result = "Normal/Benign"
        
    return result


title = "Breast cancer detection with Deep Learning (ConvNext)"
description = "<p style='text-align: center'><b>As a radiologist or oncologist, it is crucial to know what is wrong with a breast x-ray image.<b><br><b>Upload the breast X-ray image to know what is wrong with a patients breast with or without inplant<b><p>"
article="<p style='text-align: center'>Web app is built and managed by IUH team></p>"
examples = [r'C:\Users\huumi\Desktop\Subjects\HK8\DOAN_THUCTAP\BreastCancer\D1-0001_1-1.png', 
            r'C:\Users\huumi\Desktop\Subjects\HK8\DOAN_THUCTAP\BreastCancer\D1-0005_1-2.png',
            r'C:\Users\huumi\Desktop\Subjects\HK8\DOAN_THUCTAP\BreastCancer\mdb028.png']
enable_queue=True
#interpretation='default'

gr.Interface(fn=predict,
             inputs=gr.Image(label="Image (png file)"),
             outputs='text',
             title=title,
             description=description,
             article=article,
             examples=examples,
             enable_queue=enable_queue
             ).launch(share=True)
