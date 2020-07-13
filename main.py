from option import args
from sub_main import main_SRCNN, main_VDSR, main_EDSR
opt = args
opt.cuda = not opt.cpu

def main():
    global opt
    print("CHECK | runt type => ", opt.run_type)
    print("CHECK | model => ", opt.model)
    if opt.run_type == 'train':
        print("CHECK | training details [ epoch => ", opt.epochs, " ] [ scale => ", opt.scale,
              " ] [ batch size => ", opt.batch_size, " ]")
        if opt.patch: print("CHECK | [ using patch => ", opt.patch, " ] [ patch size => ", opt.patch_size , " ]")

        start = input("If you want to start training ? ( Yes : 1, No : 2) ")
        visdom = input("If you want to use visdom ? ( Yes : 1 , No : 2) ")
        use_patch = input("If you want to use patch at training ? ( Yes : 1 , No : 2) ")

        if use_patch == '1' : opt.patch = not opt.patch
        opt.use_visdom = int(visdom)
        if start == '1':
            if opt.model == 'SRCNN':
                SRCNN = main_SRCNN.mainSRCNN(opt)
                SRCNN.main()
            if opt.model == 'VDSR':
                VDSR = main_VDSR.mainVDSR(opt)
                VDSR.main()
        else: exit()
    else :
        if opt.model == 'SRCNN':
            SRCNN = main_SRCNN.mainSRCNN(opt)
            SRCNN.main()

if __name__ == "__main__":
    main()
