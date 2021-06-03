import os.path as osp
import numpy as np
import random
#import matplotlib.pyplot as plt
import collections
import torch
import torchvision.transforms as transforms
import cv2
from torch.utils import data
from PIL import Image
import glob
import yaml
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
import transformations as tr


class MVD():
    def __init__(self, root, label_mapping_config, training=True):
        self.root = root

        #map_root_image = os.path.join(self.root,'images')
        self.files = []
        self.training = training
        label_mapping_yaml = open(label_mapping_config)
        parsed_label_mapping_yaml_file = yaml.load(label_mapping_yaml, Loader=yaml.FullLoader)
        self.label_mapping = parsed_label_mapping_yaml_file["label_mapping"]


        for file_path in glob.glob(osp.join(root, 'images/*.jpg')):
            filename = osp.basename(file_path).split('.')[0]
            img_file = file_path
            label_file = osp.join(root, 'instances', filename + '.png')
            self.files.append({
                "img": img_file,
                "label": label_file,
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instance = cv2.imread(datafiles["label"], -1)
        label = instance / 256
        label = np.uint8(label)
        sample = {"image": image, "label": label}

        if self.training:
            return self.transforms_tr(sample)
        else:
            return self.transforms_valid(sample)

    def transforms_tr(self, sample):
        prob_augmentation = random.random()
        if prob_augmentation > 0.1:
            augmentation_strength = int(np.random.uniform(10, 30))
            if sample['image'].shape[0] < 360 or sample['image'].shape[1] < 480:
                composed_transforms = transforms.Compose([
                    tr.Resize((360, 480)),
                    tr.RandAugment(3, augmentation_strength),
                    tr.LabelMapping(self.label_mapping),
                    tr.ToTensor()])
            else:
                composed_transforms = transforms.Compose([
                    tr.RandomCrop((360, 480)),
                    tr.RandAugment(3, augmentation_strength),
                    tr.LabelMapping(self.label_mapping),
                    tr.ToTensor()])
        else:
            composed_transforms = transforms.Compose([
                tr.RandomCrop((360, 480)),
                tr.LabelMapping(self.label_mapping),
                tr.ToTensor()])
        return composed_transforms(sample)

    def transforms_valid(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize((360, 480)),
            tr.LabelMapping(self.label_mapping),
            tr.ToTensor()])
        return composed_transforms(sample)
if __name__ == "__main__":

    # from dataloaders import custom_transforms as tr
    # from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--label-mapping-config", type=str,
                        help="Path to the label mapping config yaml")

    args = parser.parse_args()
    train_dataset = MVD(osp.join(args.data_dir, 'training'), args.label_mapping_config)
    trainloader = data.DataLoader(train_dataset, shuffle=True, batch_size=1,
                                  num_workers=4, pin_memory=True)
    cv2.namedWindow("image")
    # cv2.namedWindow("labels")
    trainloader_enu = enumerate(trainloader)

    for step in range(10):

        batch = next(trainloader_enu)

        index, sample = batch

        label = sample["label"]
        image = sample["image"]


        # if index % 100 == 0:
        #     print('%d processd' % (index))

        # image, label, size = batch
        image = np.array(image[0]).astype(np.uint8)
        label = np.array(label[0]).astype(np.uint8)

        image = image.transpose((1, 2, 0))
        cv2.imshow("image", image)
        # # cv2.imshow("labels", label)
        #
        cv2.waitKey()