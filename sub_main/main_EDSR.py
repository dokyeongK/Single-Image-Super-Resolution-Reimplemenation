import data
import os
import torch
import torchvision.utils as utils
from sub_main import sub_main
from model import EDSR
import torch.optim as optim
import torch.nn as nn
from math import log10
import visdom
import numpy as np
import torchvision

class mainEDSR(sub_main):

    def __init__(self, args):
        self.args = args
        self.mseLoss = nn.MSELoss()
        if self.args.use_visdom == 1:
            self.visdom = visdom.Visdom()
            self.visdom_data = {'X': [], 'Y': []}
        else :
            self.visdom = None
            self.visdom_data = None
        self.lr = args.lr

    def train(self, model, optimizer, epochs, train_data_loader, test_data_loader):
        print("VDSR ==> Training .. ")
        print_loss = 0
        psnr = 0
        for epoch in range(1, epochs):
            self.lr = self.adjust_learning_rate(epoch, self.lr, self.args.lr_decay)
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr

            for itr, data in enumerate(train_data_loader):
                imgLR, imgHR = data[0], data[1]
                if self.args.cuda:
                    imgLR = imgLR.cuda()
                    imgHR = imgHR.cuda()
                result = model(imgLR)
                srcnn_loss = self.mseLoss(result, imgHR)
                optimizer.zero_grad()
                srcnn_loss.backward(retain_graph=True)
                optimizer.step()

                MSE = self.mseLoss(result, imgHR)
                psnr += 10 * log10(1 / MSE.item())

                if itr % 10 == 0:
                    print("epoch/iteration/learning-rate [", epoch, " / ", itr, " / ", self.lr,
                          "] MSE Loss => {:.6f}, AVG PSNR value=> {:.6f}".format(srcnn_loss.data, psnr / 10))
                    psnr = 0

                if itr % 10 == 0 and self.visdom is not None:
                    self.visdom_data['X'].append(print_loss)
                    self.visdom_data['Y'].append([torch.Tensor([srcnn_loss]).unsqueeze(0).cpu()])
                    self.visdom.line(
                        X=np.stack([np.array(self.visdom_data['X'])], 1),
                        Y=np.array(self.visdom_data['Y']),
                        win=1,
                        opts=dict(xlabel='Step',
                                  ylabel='Loss',
                                  title='Training loss result',
                                  legend=['MSE Loss']),
                        update='append'
                    )
                    lr_visdom = self.tensor2im(imgLR.data)
                    hr_visdom = self.tensor2im(imgHR.data)
                    result_visdom = self.tensor2im(result.data)
                    self.visdom.image(lr_visdom.transpose([2, 0, 1]), opts=dict(title='input'), win=2)
                    self.visdom.image(hr_visdom.transpose([2, 0, 1]), opts=dict(title='target'), win=3)
                    self.visdom.image(result_visdom.transpose([2, 0, 1]), opts=dict(title='result'), win=4)
                    print_loss += 1

            self.test(self.args, test_data_loader, model, epoch)
            self.save_model(model, epoch)

    def test(self, args, test_data_loader, model, epoch = 0):
        print("EDSR ==> Testing .. ")
        model.eval()
        if not os.path.exists('./test_result'): os.makedirs(os.path.join('./test_result'))
        if args.cuda:
            for iteration, batch in enumerate(test_data_loader):
                input, filename = batch[0], batch[1]
                if args.cuda: input = input.cuda()
                with torch.no_grad():
                    result = model(input)
                    if self.args.cuda:
                        result_img = result.data.cpu().squeeze(0)
                    utils.save_image(result_img,'test_result/EDSR'+filename[0]+'_{}.png'.format(epoch))

        print("EDSR ==> Saved result at [test_result] .. ")

    def adjust_learning_rate(self, epoch, lr, step):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        lr = lr * (0.1 ** (epoch // step))
        return lr

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = torchvision.utils.make_grid(image_tensor).cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
        return image_numpy.astype(imtype)

    def save_model(self, model, epoch):
        # TO-DO : 모델명!
        model_out_path = "EDSR_train_{}.pth".format(epoch)
        torch.save(model.state_dict(), './checkpoint/EDSR/' + model_out_path)
        print("Training model saved to {}".format(model_out_path))

    def main(self):
        global model
        print("EDSR ==> Data loading .. ")
        loader = data.Data(self.args)
        print("EDSR ==> Check run type .. ")
        if self.args.run_type == 'train':
            train_data_loader = loader.loader_train
            test_data_loader = loader.loader_test
            print("EDSR ==> Load model .. ")
            model = EDSR.EDSR()
            print("EDSR ==> Setting optimizer .. [ ", self.args.optimizer, " ] , lr [ ", self.args.lr, " ] , Loss [ MSE ]")
            optimizer = optim.Adam(model.parameters(), self.args.lr)
            if self.args.cuda:
                model.cuda()
            self.train(model, optimizer, self.args.epochs, train_data_loader, test_data_loader)
        elif self.args.run_type == 'test':
            print("EDSR ==> Testing .. ")
            if os.path.exists(self.args.pre_model_dir):
                if not os.path.exists(self.args.dir_data_test_lr): print("EDSR ==> Fail [ Test model is not exists ]")
                else:
                    test_data_loader = loader.loader_test
                    Loaded = torch.load(self.args.pre_model_dir)
                    model.load_state_dict(Loaded)
                    if self.args.cuda:
                        model.cuda()
                    self.test(self.args, test_data_loader, model)
            else : print("EDSR ==> Fail [ Pretrain model directory is not exists ]")
