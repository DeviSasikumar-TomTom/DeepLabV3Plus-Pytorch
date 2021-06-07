import torch.utils.data as data
import cv2
import yaml
import glob
import os.path as osp
import random
import numpy as np
import utils
import transformations
import transformations as tr
import torchvision.transforms as transforms



class Mapillary(data.Dataset):

    def __init__(self,
                 root,
                 label_map,
                 download=False,
                 training=True):

        self.root = root
        self.training = training
        self.files = []
        self.img_file = []
        self.label_file = []
        label_map_yaml = open(label_map)
        parsed_label_mapping_yaml_file = yaml.load(label_map_yaml, Loader=yaml.FullLoader)
        self.label_mapping = parsed_label_mapping_yaml_file["label_mapping"]


        for file_path in glob.glob(osp.join(root, 'images/*.jpg')):
            filename = osp.basename(file_path).split('.')[0]
            img_file = file_path
            self.img_file.append(img_file)

            image_root = osp.join(root,'images')
        for filepath1 in glob.glob(osp.join(root, 'instances/*.png')):
            filename1 = osp.basename(filepath1).split('.')[0]
            label_file = filepath1


            self.label_file.append(label_file)

            self.files.append({
                "img": img_file,
                 "label": label_file
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

       #image1 = Image.open(self.img_file[index]).convert('RGB')
       #label1 = Image.open(self.label_file[index])

        if self.training:
            return self.transforms_tr(sample)
        else:
            return self.transforms_valid(sample)



    def __len__(self):
        return len(self.img_file)

    def transforms_tr(self, sample):
        prob_augmentation = random.random()
        if prob_augmentation > 0.1:
            augmentation_strength = int(np.random.uniform(10, 30))
            if sample['image'].shape[0] < 360 or sample['image'].shape[1] < 480:

                 composed_transforms = transforms.Compose([tr.Resize((360, 480)),
                                                          tr.RandAugment(3, augmentation_strength),
                                                          tr.LabelMapping(self.label_mapping), tr.ToTensor()])

            else:
                composed_transforms = transforms.Compose([transformations.RandomCrop((360, 480)),
                                                          tr.RandAugment(3, augmentation_strength),
                                                          tr.LabelMapping(self.label_mapping),
                                                          tr.ToTensor()])
        else:
            composed_transforms = transforms.Compose([tr.RandomCrop((360, 480)),
                                                      tr.LabelMapping(self.label_mapping),
                                                      tr.ToTensor()])
        return composed_transforms(sample)

    def transforms_valid(self, sample):
        composed_transforms = transforms.Compose([tr.Resize((360, 480)),
                                                  tr.LabelMapping(self.label_mapping),
                                                  tr.ToTensor()])
        return composed_transforms(sample)

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import argparse

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
