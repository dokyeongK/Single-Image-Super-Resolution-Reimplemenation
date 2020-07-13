import os
import os.path
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor()])

class TEST(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.train = False
        self.scale = args.scale
        self.pair = os.path.exists(self.args.dir_data_test_hr)
        self._set_filesystem(args.dir_data_test_lr)
        if self.pair :
            self.images_lr, self.images_hr= self._scan()
        else :
            self.images_lr = self._scan()

    def __getitem__(self, idx):
        if self.pair:
            lr, hr, filename = self._load_file(idx)
            filename = os.path.split(self.images_lr[idx])[-1]
            filename = os.path.splitext(filename)[0]
            lr_tensor, hr_tensor = transform(lr), transform(hr)
            return lr_tensor, hr_tensor, filename
        else :
            lr, filename = self._load_file(idx)
            filename = os.path.split(self.images_lr[idx])[-1]
            filename = os.path.splitext(filename)[0]
            lr_tensor = transform(lr)
            return lr_tensor, filename

    def _set_filesystem(self, dir_data):
        self.apath = dir_data
        if self.pair: self.dir_hr = os.path.join(self.apath)
        self.dir_lr = os.path.join(self.apath)
        self.ext = '.png'

    def _load_file(self, idx):
        idx = self._get_index(idx)
        filename = self.images_lr[idx]
        lr_img = Image.open(self.images_lr[idx]).convert('RGB')
        h_size = lr_img.size[0] * self.scale
        w_size = lr_img.size[1] * self.scale
        lr_img = lr_img.resize((h_size, w_size), Image.BICUBIC)
        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        if self.pair:
            filename = self.images_hr[idx]
            hr_img = Image.open(self.images_hr[idx]).convert('RGB')
            return lr_img, hr_img, filename

        else: return lr_img, filename

    def __len__(self):
        return len(self.images_lr)

    def _scan(self):

        file_list_lr = [file for file in os.listdir(self.dir_lr) if file.endswith(".jpg") or file.endswith(".png")]
        lr_images = [os.path.join(self.dir_lr, x) for x in file_list_lr]

        if self.pair:
            file_list_hr = [file for file in os.listdir(self.dir_hr) if file.endswith(".jpg") or file.endswith(".png")]
            hr_images = [os.path.join(self.dir_hr, x) for x in file_list_hr]
            return lr_images, hr_images

        else : return lr_images

    def _get_index(self, idx):
        return idx
