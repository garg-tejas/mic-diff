import os, torch, cv2, random
import numpy as np
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
from scipy.ndimage.morphology import binary_erosion
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import filters
import numpy as np
import imageio
import dataloader.transforms as trans
import json, numbers
from glob import glob
import pickle

class MedicalPreprocessing:
    def __call__(self, img):

        img_np = np.array(img)

        if len(img_np.shape) == 3:
            green = img_np[:, :, 1]
        else:
            green = img_np
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(green)
        
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return Image.fromarray(denoised).convert('RGB')

class APTOSDataset(Dataset):
    def __init__(self, data_list, train=True, dataroot="./dataset/APTOS2019/"):
        self.trainsize = (224,224)
        self.train = train
        self.dataroot = dataroot
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                MedicalPreprocessing(),
                trans.RandomHorizontalFlip(),
                trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                MedicalPreprocessing(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = data_pac['img_root']
        if os.path.isabs(img_path):
            full_img_path = img_path
        else:
            full_img_path = os.path.join(self.dataroot, img_path)
        img = Image.open(full_img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        
        return img_torch, label

    def __len__(self):
        return self.size
