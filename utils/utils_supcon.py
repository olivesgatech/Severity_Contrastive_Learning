from torchvision import transforms, datasets
from datasets.fisheye_dataset import FisheyeFordDataset
from datasets.oct_dataset import OCTDataset
from datasets.biomarker import BiomarkerDatasetAttributes
from utils.utils import TwoCropTransform
from datasets.prime import PrimeDatasetAttributes
from datasets.prime_trex_combined import CombinedDataset
from datasets.recovery import recovery
from datasets.chest import ChexpertDataset,COVIDKaggleDataset
from datasets.trex import TREX
from datasets.gradcon_dataset import GradConDataset
from datasets.ood import OODDataset
from datasets.oct_cluster import OCTDatasetCluster
from datasets.chest_clusters import Chexpert_Clusters_Dataset
import torch
from models.resnet import SupConResNet
from loss.loss import SupConLoss
import torch.backends.cudnn as cudnn
from datasets.ford_region import FordRegion
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
import torch.nn as nn
def set_model_contrast(opt):
    if(opt.dataset == 'Ford' or opt.dataset =='Ford_Region'):
        model = SupConResNet(name=opt.model)
        model.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        print('Hello')
    else:
        model = SupConResNet(name=opt.model)

    criterion = SupConLoss(temperature=opt.temp,device=opt.device)
    print(opt.dataset)
    device = opt.device
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if opt.parallel == 1:
            model = torch.nn.DataParallel(model)
            model = model.cuda()
            criterion = criterion.cuda()
        else:
            model = model.to(device)
            criterion = criterion.to(device)
        cudnn.benchmark = True

    return model, criterion


def set_loader(opt):
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
    elif opt.dataset == 'OCT_Cluster':

        mean = (.1904)
        std = (.2088)
    elif opt.dataset == 'Prime' or opt.dataset == 'OOD' or opt.dataset == 'CombinedBio' or opt.dataset == 'CombinedBio_Modfied' or opt.dataset =='Prime_Compressed' or opt.dataset == 'GradCon' or opt.dataset == 'Grad' or opt.dataset =='GradCon_Noise':

        mean = (.1706)
        std = (.2112)
    elif opt.dataset == 'TREX_DME' or opt.dataset == 'Prime_TREX_DME_Fixed' \
            or opt.dataset == 'Prime_TREX_Alpha' or opt.dataset == 'Prime_TREX_DME_Discrete' \
            or opt.dataset == 'Patient_Split_2_Prime_TREX' or opt.dataset == 'Patient_Split_3_Prime_TREX':
        mean = (.1651)
        std = (.2118)
        #mean = 0
        #std = 1
    elif opt.dataset == 'Chexpert' or opt.dataset == 'covid_kaggle' or opt.dataset == 'Chexpert_Cluster':
        mean = (.5093)
        std = (.2534)
    elif opt.dataset == 'PrimeBio':
        print(2)
        mean = (.1706)
        std = (.2112)
    elif opt.dataset == 'Prime_Comb_Bio':
        print(3)
        mean = (.1706)
        std = (.2112)
    elif opt.dataset == 'Recovery' or opt.dataset == 'Recovery_Compressed':
        print(4)
        mean = (.1706)
        std = (.2112)
    elif opt.dataset == 'path':

        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    '''
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            normalize,
        ])
    '''
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


    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'Ford':
        df = '/home/kiran/Desktop/Dev/FishEyeFord/contrastive_testing/patches_train.csv'
        img_dir = '/data/Datasets/WoodScape/bbox_extraction_images_trainset'
        train_dataset = FisheyeFordDataset(df,img_dir,transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'Ford_Region':
        df = '/home/kiran/Desktop/Dev/FishEyeFord/contrastive_testing/patches_train_10.csv'
        img_dir = '/data/Datasets/WoodScape/Extracted_BBOX_3Classes'

        train_dataset = FordRegion(df, img_dir, transforms=TwoCropTransform(train_transform))
    elif opt.dataset =='OCT':
        csv_path_train = 'oct_files/train_csv_patients.csv'
        data_path_train ='/data/Datasets/ZhangLabData/CellData/OCT/train'
        train_dataset = OCTDataset(csv_path_train,data_path_train,transforms = TwoCropTransform(train_transform))
    elif opt.dataset == 'Chexpert':
        img_dir = '/data/kiran/'
        csv_path_train = opt.train_csv_path
        train_dataset = ChexpertDataset(csv_path_train, img_dir, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'Chexpert_Cluster':
        img_dir = '/data/Datasets/'
        csv_path_train = opt.train_csv_path
        train_dataset = Chexpert_Clusters_Dataset(csv_path_train, img_dir, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'Prime':
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/full_prime_train.csv'
        #csv_path_train = './prime/train_prime_full_train.csv'
        data_path_train = '/data/Datasets/Prime'
        train_dataset = PrimeDatasetAttributes(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'CombinedBio' or opt.dataset == 'CombinedBio_Modfied':
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/complete_biomarker_training.csv'
        data_path_train = '/data/Datasets/Prime_FULL_128'
        train_dataset = BiomarkerDatasetAttributes(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'Recovery':
        csv_path_train = '/final_csvs_1/full_recovery_train.csv'
        data_path_train = '/data/Datasets/RECOVERY'
        train_dataset = recovery(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'TREX_DME':
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/Datasets_Conjoined/trex_compressed.csv'
        data_path_train = '/data/Datasets/TREX DME'
        train_dataset = TREX(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'GradCon' or opt.dataset == 'Grad' or opt.dataset == 'GradCon_Noise':
        #if(opt.gradcon_labels == 100 and opt.patient_split == 1):
        csv_path_train = '/home/kiran/Desktop/Dev/gradcon-anomaly/prime_trex_gradient_noise_discretized.csv'
        '''
        elif(opt.gradcon_labels == 5000 and opt.patient_split == 1):
            csv_path_train = '/home/kiran/Desktop/Dev/gradcon-anomaly/datasets/patient_split_1/prime_trex_gradcon_process_5000_ps1.csv'
        elif(opt.gradcon_labels == 1000):
            csv_path_train = '/home/kiran/Desktop/Dev/SupCon/gradcon_files/prime_trex_gradcon_process_1000.csv'
        elif (opt.gradcon_labels == 50):
            csv_path_train = '/home/kiran/Desktop/Dev/SupCon/gradcon_files/prime_trex_gradcon_process_50.csv'
        elif (opt.gradcon_labels == 500):
            csv_path_train = '/home/kiran/Desktop/Dev/SupCon/gradcon_files/prime_trex_gradcon_process_500.csv'
        elif(opt.gradcon_labels == 2000):
            csv_path_train = '/home/kiran/Desktop/Dev/SupCon/gradcon_files/prime_trex_gradcon_process_2000.csv'
        elif(opt.gradcon_labels == 5000):
            csv_path_train = '/home/kiran/Desktop/Dev/gradcon-anomaly/datasets/patient_split_3/prime_trex_gradcon_process_5000.csv'
        elif (opt.gradcon_labels == 10000):
            csv_path_train = '/home/kiran/Desktop/Dev/gradcon-anomaly/datasets/prime_trex_gradcon_process_10000.csv'
        '''
        data_path_train = '/data/Datasets/TREX DME'
        train_dataset = GradConDataset(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'OOD':
        csv_path_train = '/home/kiran/Desktop/Dev/SupCon/discretized_prime_trex_mahal_correct_ood.csv'
        data_path_train = '/data/Datasets/TREX DME'
        train_dataset = OODDataset(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'TREX_DME_Recovery':
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/Datasets_Conjoined/trex_compressed.csv'
        data_path_train = '/data/Datasets/TREX DME'
        train_dataset = CombinedDataset(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'Prime_Recovery':
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/Datasets_Conjoined/trex_compressed.csv'
        data_path_train = '/data/Datasets/TREX DME'
        train_dataset = CombinedDataset(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'Prime_Recovery_TREX_DME':
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/Datasets_Conjoined/trex_compressed.csv'
        data_path_train = '/data/Datasets/TREX DME'
        train_dataset = CombinedDataset(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'Prime_TREX_DME_Fixed' or opt.dataset == 'Prime_TREX_Alpha' \
            or opt.dataset == 'Patient_Split_2_Prime_TREX' or opt.dataset == 'Patient_Split_3_Prime_TREX':
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/Datasets_Conjoined/prime_trex_compressed.csv'
        data_path_train = '/data/Datasets/TREX DME'
        train_dataset = CombinedDataset(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'Prime_TREX_DME_Discrete':
        csv_path_train = './final_csvs_' + str(opt.patient_split) + '/Discretized_Datasets/cuts_' + str(opt.discrete_level) + ".csv"
        data_path_train = '/data/Datasets/TREX DME'
        train_dataset = CombinedDataset(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'Prime_Compressed':
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/Datasets_Conjoined/prime_compressed.csv'
        data_path_train = '/data/Datasets/Prime'
        train_dataset = CombinedDataset(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'Recovery_Compressed':
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/Datasets_Conjoined/recovery_compressed.csv'
        data_path_train = '/data/Datasets/Prime'
        train_dataset = CombinedDataset(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'Prime_Percentages':
        csv_path_train = 'percentage_removed_labels/train_prime_' + str(opt.percentage) + '.csv'
        data_path_train = '/data/Datasets/Prime_FULL_128'
        train_dataset = PrimeDatasetAttributes(csv_path_train, data_path_train,
                                                    transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'Recovery_Percentages':
        csv_path_train = 'percentage_removed_labels/train_recovery_' + str(opt.percentage) + '.csv'
        data_path_train = '/data/Datasets/Prime_FULL_128'
        train_dataset = recovery(csv_path_train, data_path_train,
                                               transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'OCT_Cluster':
        csv_path_train = 'oct_files/train_1000.csv'
        data_path_train = '/data/Datasets/ZhangLabData/CellData/OCT/train'
        train_dataset = OCTDatasetCluster(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,drop_last=True)

    return train_loader


def set_model(opt):
    if(opt.dataset == 'Ford'):
        model = SupConResNet(name=opt.model)
    else:
        model = SupConResNet(name=opt.model)

    criterion = SupConLoss(temperature=opt.temp,device=opt.device)
    print(model)
    device = opt.device
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if opt.parallel == 1:
            model = torch.nn.DataParallel(model)
            model = model.cuda()
            criterion = criterion.cuda()
        else:
            model = model.to(device)
            criterion = criterion.to(device)
        cudnn.benchmark = True

    return model, criterion