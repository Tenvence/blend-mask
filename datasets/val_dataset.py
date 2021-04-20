import os

import albumentations as alb
import torchvision.datasets as cv_datasets

import datasets.tools as tools


class ValDataset(cv_datasets.CocoDetection):
    def __init__(self, root, year, input_size):
        super(ValDataset, self).__init__(root=os.path.join(root, f'val{year}'), annFile=os.path.join(root, 'annotations', f'instances_val{year}.json'))
        self.h, self.w = input_size
        self.img_transform = alb.Compose([alb.Resize(width=self.w, height=self.h)])

        self.points, _ = tools.encode_points_and_regress_ranges(self.h, self.w)

    def __getitem__(self, index):
        img, _ = tools.load_img_target(self, index)

        img = self.img_transform(image=img)['image']
        img = tools.TENSOR_TRANSFORM(img)
        img_id = self.ids[index]

        return img, self.points, img_id
