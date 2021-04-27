import os
import torch
from torch.utils import data
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def get_dataset(path, phase):
    # if phase == "train":
    dataset = ImageDataset(path, phase)

    # else:
    #     dataset = None
    #     print("Invalid phase of dataset")

    return dataset


class ImageDataset(data.Dataset):
    def __init__(self, data_path, phase='train'):
        self.phase = phase
        self.data_path = data_path
        self.len = len(os.listdir(self.data_path))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform_uncert = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return self.len

    def plt_img(self, tensor):
        plt.imshow(tensor.permute(1, 2, 0))
        plt.show()

    def __getitem__(self, index):
        file_path = os.path.join(self.data_path, f"{index}")
        img_og = Image.open(os.path.join(file_path, "img_og.png"))
        semantic = Image.open(os.path.join(file_path, "semantic.png")).convert('RGB')
        depth_pred = Image.open(os.path.join(file_path, "depth_pred_.png")).convert('RGB')
        normal = Image.open(os.path.join(file_path, "normal_pred.png")).convert('RGB')

        sample = {
            'depth_pred': depth_pred,
            'semantic_pred': semantic,
            'normal_pred': normal,
            'img_og': img_og
        }
        if self.transform:
            # apply the transform of the images
            sample = {k: self.transform(v) for k, v in sample.items()}

        if self.phase == 'train' or self.phase == 'test_demo':
            depth_uncert = Image.open(os.path.join(file_path, "depth_uncertainty.png")).convert('L')
            semantic_uncert = Image.open(os.path.join(file_path, "semantic_uncertainty.png")).convert('L')
            normal_uncert = Image.open(os.path.join(file_path, "normal_uncert.png")).convert('L')
            sample['depth_uncert'] = self.transform_uncert(depth_uncert)
            sample['semantic_uncert'] = self.transform_uncert(semantic_uncert)
            sample['normal_uncert'] = self.transform_uncert(normal_uncert)
            resize_transform = torchvision.transforms.Resize((224, 224))
            sample["depth_rse"] = torch.from_numpy(np.load(os.path.join(file_path, "depth_rse.npz"))['arr_0'])
            sample["depth_rse"] = resize_transform(sample["depth_rse"].unsqueeze(0)).float()
            sample["semantic_ce"] = torch.from_numpy(np.load(os.path.join(file_path, "seg_ce.npz"))['arr_0'])
            sample["semantic_ce"] = resize_transform(sample["semantic_ce"]).float()
            sample["semantic_ce"] = (sample["semantic_ce"] - sample["semantic_ce"].min())/((sample["semantic_ce"].max() - sample["semantic_ce"].min()))
            sample["depth_rse"] = (sample["depth_rse"] - sample["depth_rse"].min())/((sample["depth_rse"].max() - sample["depth_rse"].min()))
        sample["img_og"] = self.transform(img_og)
        # print(list(np.load(os.path.join(file_path, "depth_rse.npz")).keys()))


        return sample
