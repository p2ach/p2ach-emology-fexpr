import os
import numpy as np
# import pandas as pd
import csv
# import pandas as pd
# from torchvision.io import read_image

from torch.utils import data
from torchvision import transforms

train_transform=transforms.Compose([
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])

validation_transform=transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])


class FaceExpr(data.Dataset):
    def __init__(self, src_dir, transform=None, target_transform=None):
        self.img_dir = os.path.join(src_dir,'images')
        self.lbl_dir = os.path.join(src_dir,'labels')
        self.imgs = os.listdir(self.img_dir)
        # self.lbls = os.listdir(os.path.join(src_dir,'labels'))
        # self.img_labels = pd.read_csv(annotations_file)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        lbl_path = os.path.join(self.lbl_dir, self.imgs[idx].split('.')[0]+'.csv')



        image = np.load(img_path)
        with open(lbl_path, 'r') as csvfile:
            reader = csvfile.read()
            lbl=reader.split('\n')[0].split(',')
                # lbl=row
                # break
            # row=reader[0]
        lbl=lbl[1:]
        lbl = [float(l) for l in lbl]

        lbl=np.array(lbl,dtype=np.float32)
        label=np.argmax(lbl)
        # print("row", label)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label