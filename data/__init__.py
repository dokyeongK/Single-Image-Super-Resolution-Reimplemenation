from importlib import import_module
from torch.utils.data import DataLoader

class Data:
    def __init__(self, args):
        if args.run_type == 'train':
            module_train = import_module('data.dataset_' + args.model)
            trainset = getattr(module_train, args.model)(args) #개체의 속성을 가져옴.
            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=False,
            )

        module_test = import_module('data.dataset_TEST')
        testset = getattr(module_test, args.test_dataset)(args)
        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
        )


