import argparse

parser = argparse.ArgumentParser(description='INTERGRATE _ SUPER RESOLUTION')

# PATH SETTING : dataset, pre_train model, save test|training result => if not exit :: make
parser.add_argument('--pre_model_dir', type=str, default='./pretrain_model/EDSR/',
                    help='model directory')
parser.add_argument('--dir_data_train', type=str, default='./dataset/train/',
                    help='dataset directory')
parser.add_argument('--dir_data_test_lr', type=str, default='./dataset/test/LR/Set5/',
                    help='dataset directory')
parser.add_argument('--dir_data_test_hr', type=str, default='NONE',
                    help='dataset directory')

# DATA PREPROCESSING OPTION : dataset type(DIV2K), chnnel type( GRAY | RGB | LAB ), patch use, augment ( 90ROTATE | HREVERSE | WREVERSE ), noise
parser.add_argument('--train_dataset', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--test_dataset', type=str, default='TEST',
                    help='train dataset name')
parser.add_argument('--channel_type', type=str, default='RGB',
                    help='channel type to train or test ( GRAY | RGB | LAB )')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--patch', action='store_true',
                    help='enables training using patch')
parser.add_argument('--patch_size', type=int, default=128,
                    help='output patch size')
parser.add_argument('--train_size', type=int, default=196,
                    help='output patch size')
parser.add_argument('--augment_type', type=int, default=0,
                    help='augment type to train or test if you want to use  -> 0 dont want -> 1')
parser.add_argument('--noise', type=str, default='.',
                    help='Gaussian noise std.')
parser.add_argument('--augment_rotate', type=int, default=0,
                    help='rotate option [ Yes -> 0, No -> 1')
parser.add_argument('--augment_T2B', type=int, default=0,
                    help='rotate option (flip Top to bottom) [ Yes -> 0, No -> 1')
parser.add_argument('--augment_L2R', type=int, default=0,
                    help='rotate option (flip Left to Right) [ Yes -> 0, No -> 1')

# BASIC OPTION : run type, scale factor, model type, run type( TEST | TRAINING )
parser.add_argument('--run_type', type=str, default='train',
                    help='train | test')
parser.add_argument('--use_visdom', type=int, default=1,
                    help='use -> 1, not use -> 2')
parser.add_argument('--scale', type = int, default='4',
                    help='super resolution scale')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--model', default='EDSR',
                    help='model name')
parser.add_argument('--ext', type=str, default='img',
                    help='dataset file extension')

# Model specifications
parser.add_argument('--pre_train', type=int, default=1,
                    help='0 : use pre-trained model, 1 : dont use pre-trained model')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')

# TRAINING OPTION : epoch, batch size, use_cuda
parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--cpu', action='store_true',
                    help='enables CUDA training')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=10,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=int, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')

args = parser.parse_args()

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False