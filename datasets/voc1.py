import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


class Mapillary(data.Dataset):

    def __init__(self,
                 root,
                 label_map,
                 download=False,
                 transform=None):

        is_aug = False
        if year == '2012_aug':
            is_aug = True
            year = '2012'

        self.root = os.path.expanduser(root)
        label_map_yaml = open(label_map)
        parsed_label_mapping_yaml_file = yaml.load(label_map_yaml, Loader=yaml.FullLoader)
        self.label_mapping = parsed_label_mapping_yaml_file["label_mapping"]

        self.transform = transform

        self.image_set = image_set
        #base_dir = DATASET_YEAR_DICT[year]['base_dir']
        #voc_root = os.path.join(self.root, base_dir)
        #image_dir = os.path.join(voc_root, 'JPEGImages')

        for file_path in glob.glob(osp.join(root, 'images/*.jpg')):
            filename = osp.basename(file_path).split('.')[0]
            img_file = file_path
            label_file = osp.join(root, 'instances', filename + '.png')
            self.files.append({
                "img": img_file
                "label": label_file
            })
    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instance = cv2.imread(datafiles["label"], -1)
        label = instance/256
        label = np.uint8(label)
        sample = {"image": image, "label": label}

        if self.training:
            return self.transforms_tr(sample)
        else:
            return self.transforms_valid(sample)

    def transforms_tr(self, sample):
        prob_augmentation = random.random()
        if prob_augmentation > 0.1:
            augmentation_strength = int(np.random.uniform(10,30))
            if sample['image'].shape[0] < 360 or sample['image'].shape[1] < 480:
                composed_transforms = transforms.Compose([tr.Resize((360,480)),
                                                          tr.RandAugment(3, augmentation_strength),
                                                          tr.LabelMapping(self.label_mapping),
                                                          tr.ToTensor()])
            else:
                composed_transforms = transforms. Compose([tr.RandomCrop((360,480)),
                                                           tr.RandAugment(3, augmentation_strength),
                                                           tr.LabelMapping(self.label_mapping),
                                                           tr.ToTensor()])
        else:
            composed_transforms = transforms.Compose([tr.RandomCrop((360,480)),
                                                      tr.LabelMapping(self.label_mapping),
                                                      tr.ToTensor()])
        return composed_transforms(sample)


    def transforms_valid(self,sample):
        composed_transforms = transforms.Compose([tr.Resize((360,480)),
                                                  tr.LabelMapping(self.label_mapping),
                                                  tr.ToTensor()])
        return composed_transforms(sample)


if __name__ = "__main2__":

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type = str, help = "Path to the directory containing dataset")
    parser.add_argument("--label-mapping-config", type = str, help = "path to label mapping config yaml")

    args = parser.parse_args()

    train_dataset = Mapillary(osp.join(args.data_dir, 'training'), args.label_mapping_config)
    trainloader = data.DataLoader(train_dataset, shuffle='True', batch_size=1,num_workers=4,pin_memory=True)

    cv2.namedWindow("image")

    trainloader_enu = enumerate(trainloader)

    for step in range(10):

        batch = next(trainloader_enu)

        index, sample = batch
        label = sample["label"]
        image = sample["image"]



        image = np.array(image[0]).astype(np.uint8)
        label = np.array(label[0]).astype(np.uint8)

        image = image.transpose((1,2,0))

        cv2.imshow("image", image)

        cv2.waitKey()