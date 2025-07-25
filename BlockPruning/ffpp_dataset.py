from os import listdir
from os.path import exists, isdir, join
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import cv2
import numpy as np
import os
import random
import glob
random.seed(42)

class FaceForensicsDataset(Dataset):
    def __init__(self, dataset_path, split, resolution=224, norm_mean=IMAGENET_DEFAULT_MEAN, norm_std=IMAGENET_DEFAULT_STD):
        assert isdir(dataset_path), f"got {dataset_path}"
        self.dataset_path = dataset_path
        assert split in {"train", "val", "test"}, f"got {split}"
        self.split = split

        # parses metas from the datasets
        self.items = self.parse_datasets()
        
        # sets up the preprocessing options
        assert isinstance(resolution, int) and resolution >= 1, f"got {resolution}"
        self.resolution = resolution
        assert len(norm_mean) == 3
        self.norm_mean = norm_mean
        assert len(norm_std) == 3
        self.norm_std = norm_std

    def parse_datasets(self):
        file_path = "/ssd2/DeepFakes_may/FF++/" + self.split + ".txt"
        with open(file_path, "r") as f:
            lines = f.read()
            video_list = [[i.split(' ')[0], i.split(' ')[1]] for i in lines.split('\n')]
            random.shuffle(video_list)

            image_list = []

            for video in video_list:
                img_path = os.path.join(self.dataset_path, video[0], '*.png')
                # print('img_path:',img_path)
                img_list = glob.glob(img_path)
                for i in img_list:
                    image_list.append([i, video[1]])

            random.shuffle(image_list)
            print('init:',len(image_list))

        items = [{"image_path": img_path, "is_real": True if label=='0' else False} for img_path, label in image_list]

        return items

    def __len__(self):
        return len(self.items)

    def read_image(self, path):
        image = Image.open(path).convert('RGB')
        image = T.Compose([
            T.Resize(self.resolution + self.resolution // 8, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(self.resolution),
            T.ToTensor(),
            # T.Normalize(mean=self.norm_mean, std=self.norm_std),
        ])(image)
        return image
    
    def __getitem__(self, i):
        sample = {
            "image_path": self.items[i]["image_path"],
            "image": self.read_image(self.items[i]["image_path"]),
            "is_real": torch.as_tensor([1 if self.items[i]["is_real"] is True else 0]),
        }
        return sample
    
    @staticmethod
    def _plot_image(image):
        import matplotlib.pyplot as plt
        import einops
        plt.imshow(einops.rearrange(image, "c h w -> h w c"))
        plt.show()
        plt.close()
        
    def _plot_labels_distribution(self, save_path=None):
        import matplotlib.pyplot as plt
        
        # Count the occurrences of each label
        label_counts = {"Real": 0, "Fake": 0}
        for item in self.items:
            if item["is_real"]:
                label_counts["Real"] += 1
            else:
                label_counts["Fake"] += 1
        
        # Data for plotting
        labels = list(label_counts.keys())
        counts = list(label_counts.values())
        
        # Creating the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts, color=['blue', 'orange'])
        
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title(f'[DFFD] Distribution of labels for split {self.split}')
        plt.xticks(labels)
        plt.yticks(range(0, max(counts) + 1, max(counts) // 10))
        
        for i, v in enumerate(counts):
            plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()



if __name__=="__main__":
    import random
    dataset_path = "/hdd1/DeepFakes_may/FF++/c23_crop_face/"

    for split in {"train", "val", "test"}:
        dataset = FaceForensicsDataset(dataset_path=dataset_path, split=split, resolution=224)
        print(f"sample keys:", {k: (type(v) if not isinstance(v, torch.Tensor) else v.shape) for k, v in dataset[0].items()})
        print(f"sample keys:", {k: v for k, v in dataset[0].items()})
        dataset._plot_image(dataset[random.randint(0, len(dataset))]["image"])
        dataset._plot_labels_distribution(save_path=f"_{split}_labels_ff++.png")