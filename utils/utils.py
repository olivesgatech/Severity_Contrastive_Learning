from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from models.resnet import  SupConResNet,LinearClassifier,LinearClassifier_MultiLabel
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from datasets.fisheye_dataset import FisheyeFordDataset
from datasets.oct_dataset import OCTDataset
from datasets.biomarker import BiomarkerDatasetAttributes
from datasets.biomarker_multi import BiomarkerDatasetAttributes_MultiLabel
from datasets.biomarker_fusion import BiomarkerDatasetAttributes_Fusion
from datasets.biomarker_multi_fusion import BiomarkerDatasetAttributes_MultiLabel_MultiClass
from datasets.ford_region import FordRegion
from datasets.chest import ChexpertDataset,COVIDKaggleDataset
import torch.nn as nn
def set_model(opt):
    if(opt.dataset == 'covid_kaggle'):
        model = SupConResNet(name=opt.model)
        criterion = torch.nn.CrossEntropyLoss()

        classifier = LinearClassifier(name=opt.model, num_classes=4)
    else:
        if(opt.multi == 0 and opt.dataset !='Ford' and opt.dataset !='Ford_Region'):
            model = SupConResNet(name=opt.model)
            criterion = torch.nn.CrossEntropyLoss()

            classifier = LinearClassifier(name=opt.model, num_classes=2)
        elif(opt.dataset == 'Ford' or opt.dataset == 'Ford_Region'):
            print('Hello')
            model = SupConResNet(name=opt.model)
            model.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            criterion = torch.nn.CrossEntropyLoss()
            if(opt.ford_region == 1):
                classifier = LinearClassifier(name=opt.model, num_classes=6)
            else:
                classifier = LinearClassifier(name=opt.model, num_classes=3)
        elif(opt.multi == 1 and opt.super == 3):
            model = SupConResNet(name=opt.model)
            criterion = torch.nn.BCELoss()
            classifier = LinearClassifier(name=opt.model, num_classes=1)
        elif(opt.multi == 1):
            model = SupConResNet(name=opt.model)
            criterion = torch.nn.BCELoss()

            classifier = LinearClassifier_MultiLabel(name=opt.model, num_classes=5)
    print(opt.ckpt)
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']
    device = opt.device
    if torch.cuda.is_available():
        if opt.parallel == 0:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.to(device)
        classifier = classifier.to(device)
        criterion = criterion.to(device)
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion

def set_loader_new(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100' or opt.dataset == 'Ford' or opt.dataset == 'Ford_Region':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'OCT':
        mean = (.1904)
        std = (.2088)
    elif opt.dataset == 'Chexpert' or opt.dataset == 'covid_kaggle':
        mean = (.5093)
        std = (.2534)
    elif opt.dataset == 'Prime':
        mean = (.1706)
        std = (.2112)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        #transforms.Resize(size=(224,224)),
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),

        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == 'Ford':
        df = '/home/kiran/Desktop/Dev/FishEyeFord/contrastive_testing/patches_train.csv'
        img_dir = '/data/Datasets/WoodScape/bbox_extraction_images_trainset'
        df_test = '/home/kiran/Desktop/Dev/FishEyeFord/contrastive_testing/patches_test.csv'
        img_dir_test ='/data/Datasets/WoodScape/bbox_extraction_images_testset'
        train_dataset = FisheyeFordDataset(df,img_dir,transforms=train_transform)
        test_dataset = FisheyeFordDataset(df_test,img_dir_test,transforms=val_transform)

    elif opt.dataset == 'Ford_Region':
        df = '/home/kiran/Desktop/Dev/FishEyeFord/contrastive_testing/patches_train.csv'
        img_dir = '/data/Datasets/WoodScape/Extracted_BBOX_3Classes'
        df_test = '/home/kiran/Desktop/Dev/FishEyeFord/contrastive_testing/patches_test.csv'
        img_dir_test ='/data/Datasets/WoodScape/Extracted_BBOX_3Classes'
        train_dataset = FordRegion(df,img_dir,transforms=train_transform)
        test_dataset = FordRegion(df_test,img_dir_test,transforms=val_transform)
    elif opt.dataset == 'covid_kaggle':
        img_dir = opt.img_dir #'/data/Datasets/COVID-19_Radiography_Dataset'
        train_csv_path = opt.train_csv_path #'/home/kiran/Desktop/Dev/ECE6780_MedEmbeddings01/csv_files/COVID_KAGGLE/train.csv'
        test_csv_path =  opt.test_csv_path #'/home/kiran/Desktop/Dev/ECE6780_MedEmbeddings01/csv_files/COVID_KAGGLE/test.csv'
        train_dataset = COVIDKaggleDataset(train_csv_path, img_dir, train_transform)
        test_dataset = COVIDKaggleDataset(test_csv_path, img_dir, val_transform)
    elif opt.dataset =='OCT':
        csv_path_train = 'oct_files/train_csv_patients.csv'
        csv_path_test = 'oct_files/test_csv_patients.csv'
        data_path_train ='/data/Datasets/ZhangLabData/CellData/OCT/train'
        data_path_test ='/data//Datasets/ZhangLabData/CellData/OCT/test'
        train_dataset = OCTDataset(csv_path_train,data_path_train,transforms = train_transform)
        test_dataset = OCTDataset(csv_path_test,data_path_test,transforms = train_transform)
        #val_dataset = OCTDataset(csv_path_test,data_path_test,transforms = val_transform)

    elif opt.dataset =='Prime':
        data_path_train = '/data/Datasets/Prime'
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/complete_biomarker_training.csv'
        if (opt.biomarker == 'vit_deb'):
            csv_path_test = './final_csvs_' + str(opt.patient_split) + '/test_biomarker_sets/test_VD.csv'
        elif (opt.biomarker == 'ir_hrf'):
            csv_path_test = './final_csvs_' + str(opt.patient_split) + '/test_biomarker_sets/test_IRHRF.csv'
        elif (opt.biomarker == 'full_vit'):
            csv_path_test = './final_csvs_' + str(opt.patient_split) +'/test_biomarker_sets/test_FAVF.csv'
        elif (opt.biomarker == 'partial_vit'):
            csv_path_test = './final_csvs_'+ str(opt.patient_split) +'/test_biomarker_sets/test_PAVF.csv'
        elif (opt.biomarker == 'drt'):
            csv_path_test = './final_csvs_'+ str(opt.patient_split) +'/test_biomarker_sets/test_DRT_ME.csv'
        elif(opt.multi == 1):
            csv_path_test = './final_csvs_'+ str(opt.patient_split) +'/complete_biomarker_test.csv'
        else:
            csv_path_test = './final_csvs_' + str(opt.patient_split) + '/test_biomarker_sets/test_fluirf.csv'

        data_path_test = ""
        print(csv_path_test)
        if(opt.super == 2 and opt.multi == 0):
            print('Hi')
            train_dataset = BiomarkerDatasetAttributes_Fusion(csv_path_train, data_path_train, transforms=train_transform)
            test_dataset = BiomarkerDatasetAttributes_Fusion(csv_path_test, data_path_test, transforms=val_transform)
        elif(opt.super == 2 and opt.multi == 1):
            train_dataset = BiomarkerDatasetAttributes_MultiLabel_MultiClass(csv_path_train, data_path_train, transforms=train_transform)
            test_dataset = BiomarkerDatasetAttributes_MultiLabel_MultiClass(csv_path_test, data_path_test, transforms=val_transform)
        elif(opt.multi == 1):
            train_dataset = BiomarkerDatasetAttributes_MultiLabel(csv_path_train, data_path_train,
                                                              transforms=train_transform)
            test_dataset = BiomarkerDatasetAttributes_MultiLabel(csv_path_test, data_path_test, transforms=val_transform)
        else:
            train_dataset = BiomarkerDatasetAttributes(csv_path_train,data_path_train,transforms = train_transform)
            test_dataset = BiomarkerDatasetAttributes(csv_path_test,data_path_test,transforms = val_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)
    if(opt.biomarker == 'drt' and opt.patient_split == 1):
        dl = True
    elif(opt.multi == 1):
        dl = True
    else:
        dl=False
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10, shuffle=True,
        num_workers=0, pin_memory=True,drop_last=dl)

    return train_loader, test_loader

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):

    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def accuracy_multilabel(output,target):
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    r = roc_auc_score(target,output,multi_class='ovr')
    print(r)