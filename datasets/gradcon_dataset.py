import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import pandas as pd
import torch


class GradConDataset(data.Dataset):
    def __init__(self,df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.df.iloc[idx,0]).convert("L")
        #print(self.df.iloc[idx,0])
        image = np.array(image)
        #image = image[0:496, 504:1008]
        image = Image.fromarray(image)
        image = self.transforms(image)

        bcva=self.df.iloc[idx,1]

        cst = self.df.iloc[idx,2]
        eye_id = self.df.iloc[idx, 3]
        patient = self.df.iloc[idx,4]
        grad_label = self.df.iloc[idx,6]
        grad_1000 = self.df.iloc[idx,8]
        grad_5000 = self.df.iloc[idx,9]
        grad_10000 = self.df.iloc[idx,10]

        return image, bcva,cst,eye_id,patient,grad_label,grad_1000,grad_5000,grad_10000