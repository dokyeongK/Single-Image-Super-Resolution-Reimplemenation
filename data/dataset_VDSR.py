import os
import random
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ]
)

class VDSR(data.Dataset):
    def __init__(self, args):
        self.args = args
        self._set_filesystem(args.dir_data_train)
        self.run_type = self.args.run_type
        self.use_patch = self.args.patch  # default : true
        self.images_hr, self.images_lr = self._scan()
        if self.args.channel_type == 'RGB':
            self.img_channel = 3
        elif self.args.channel_type == 'GRAY':
            self.img_channel = 1

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        if self.use_patch : lr, hr = self._get_patch(lr, hr)
        else : lr, hr = self._get_image(lr, hr)
        if self.args.augment_type == 0 : lr, hr = augment(lr, hr, self.args)
        lr_tensor, hr_tensor = transform(lr), transform(hr)
        return lr_tensor, hr_tensor, filename

    def __len__(self):
        return len(self.images_hr)

    def _scan(self):
        file_list_hr = [file for file in os.listdir(self.dir_hr) if file.endswith(".jpg") or file.endswith(".png")]
        hr_images = [os.path.join(self.dir_hr, x) for x in file_list_hr]
        file_list_lr = [file for file in os.listdir(self.dir_lr) if file.endswith(".jpg") or file.endswith(".png")]
        lr_images = [os.path.join(self.dir_lr, x) for x in file_list_lr]
        return hr_images, lr_images

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + self.args.train_dataset
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR/')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic/X{}/'.format(self.args.scale))
        self.ext = '.png'

    def _load_file(self, idx):
        idx = self._get_index(idx)
        filename = self.images_hr[idx]
        lr_img = Image.open(self.images_lr[idx]).convert('RGB')
        hr_img = Image.open(self.images_hr[idx]).convert('RGB')
        lr_img = lr_img.resize((hr_img.size[0], hr_img.size[1])) # 미리 4배 키워줌
        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return lr_img, hr_img, filename

    def _get_patch(self, lr, hr):
        scale = self.args.scale
        patch_size = self.args.patch_size
        if self.args.run_type == 'train':
            lr, hr = get_patch(
                lr, hr, patch_size, scale
            )
        else:
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]
        return lr, hr

    def _get_image(self, lr, hr):
        scale = self.args.scale
        img_size = self.args.train_size
        lr, hr = get_image(
            lr, hr, img_size, scale
        )
        return lr, hr

    def _get_index(self, idx):
        return idx

def get_patch(img_in, img_tar, patch_size, scale):
    ih, iw = img_in.size[1], img_in.size[0] #img_in.shape[:2]  # img_in1로 바꿔야함

    p = scale
    tp = p * patch_size
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    area = (ix, iy, ix+ip, iy+ip) # 가로시작점, 세로시작점, 가로범위, 세로범위

    img_in = img_in.crop(area)
    img_tar = img_tar.crop(area)

    return img_in, img_tar

def get_image(img_in, img_tar, img_size, scale):
    crop = transforms.CenterCrop(
        (img_size,img_size))
    img_in = crop(img_in)
    img_tar = crop(img_tar)
    return img_in, img_tar

def augment(lr, hr, args):
    rotate = args.augment_rotate == 0 and random.random() < 0.5
    augment_T2B = args.augment_T2B == 0 and random.random() < 0.5
    augment_L2R = args.augment_L2R == 0 and random.random() < 0.5

    if rotate :
        lr = lr.rotate(90)
        hr = hr.rotate(90)
    if augment_T2B:
        lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
        hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
    if augment_L2R:
        lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
        hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
    return lr, hr