import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms

class ToTensor(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['gt']
        img = np.array(img)
        mask = np.array(mask)
        img = np.array(img).astype(np.float32).transpose((2, 0, 1)) / 255.0
        mask = np.array(mask).astype(np.float32) / 255.0

        return {'image': img, 'gt': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['gt']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img, 'gt': mask}

class RandomVerticalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['gt']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': img, 'gt': mask}

class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    def __call__(self, sample):
        img = sample['image']
        mask = sample['gt']
        if random.random() < 0.75:
            rotate_degree = random.choice(self.degree)
            img = img.transpose(rotate_degree)
            mask = mask.transpose(rotate_degree)

        return {'image': img, 'gt': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['gt']

        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img, 'gt': mask}



class Resize(object):
    def __init__(self, size):
        self.size = size  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['gt']
        edge = sample['edge']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        edge = edge.resize(self.size, Image.NEAREST)

        return {'image': img,
                'gt': mask,
                'edge': edge}

class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['gt']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0,2)))

        return {'image': img, 'gt': mask}

class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['gt']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'gt': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['gt']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img, 'gt': mask}

train_transforms = transforms.Compose([
            FixedResize(256),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomFixRotate(),
            RandomGaussianBlur(),
            ToTensor()])


test_transforms = transforms.Compose([
            FixedResize(256),
            ToTensor()])
