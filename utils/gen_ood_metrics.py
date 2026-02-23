import torch
from utils import AverageMeter,warmup_learning_rate,accuracy,save_model
from torchvision import transforms, datasets
import sys
import time
import numpy as np
from sklearn.metrics import precision_score,recall_score
from config.config_linear import parse_option
from utils import set_loader_new, set_model, set_optimizer, adjust_learning_rate
from models.resnet import SupCEResNet
from models.resnet import  SupConResNet,LinearClassifier,LinearClassifier_MultiLabel,SupCEResNet
from models.resnet_ood import SupCEResNet_OOD
from datasets.prime_trex_combined import CombinedDataset
from datasets.biomarker import BiomarkerDatasetAttributes
from torch.autograd import Variable
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
def main():
    opt=parse_option()
    file_ckpt = '/home/kiran/Desktop/Dev/SupCon/save/filefull_vit.pth'
    ckpt = torch.load(file_ckpt, map_location='cpu')
    state_dict = ckpt['model']
    device = opt.device
    model = SupCEResNet_OOD(name='resnet18', num_classes=2)
    csv_test = '/home/kiran/Desktop/Dev/SupCon/final_csvs_1/Datasets_Conjoined/prime_trex_compressed.csv'
    df = pd.read_csv(csv_test)
    df['MSP'] = ""
    df['ODIN'] = ""
    df['Mahal'] = ""
    csv_train = '/home/kiran/Desktop/Dev/SupCon/final_csvs_1/complete_biomarker_training.csv'
    img_dir = ''
    mean = (.1706)
    std = (.2112)
    criterion = torch.nn.CrossEntropyLoss()
    normalize = transforms.Normalize(mean=mean, std=std)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    dataset_train = BiomarkerDatasetAttributes(csv_train,img_dir,val_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)
    dataset_test = CombinedDataset(csv_test,img_dir,val_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)
    #https://github.com/facebookresearch/odin/blob/main/code/calData.py
    #https://github.com/pokaxpoka/deep_Mahalanobis_detector/blob/master/OOD_Generate_Mahalanobis.py
    if torch.cuda.is_available():
        if opt.parallel == 0:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model = model.to(device)
    temp_x = torch.rand(2, 1, 224, 224).cuda()
    temp_x = Variable(temp_x)
    temp_list = model(temp_x)[1]
    print(len(temp_list))
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    # print(feature_list[1])
    count = 0

    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
    sample_mean, precision = sample_estimator(model, 2, feature_list, train_loader)
    label_list = []
    for i in range(num_output):
        M_in = get_Mahalanobis_score(model, train_loader, 2,'/home/kiran/Desktop/Dev/SupCon/utils/output' , True, 'resnet', sample_mean, precision, i, .0014)
        M_in = np.asarray(M_in, dtype=np.float32)
        print(M_in.shape)
        if i == 0:
            Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
        else:
            Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
    print(Mahalanobis_in.shape)
    for data, vit_deb, ir_hrf, full_vit, partial_vit, fluid_irf, drt, eye_id, bcva, cst, patient in tqdm(train_loader):
        label_list.append(full_vit)
    label_array = np.asarray(label_list, dtype=np.float32)
    print(label_array.shape)
    lr = LogisticRegressionCV(n_jobs=-1).fit(Mahalanobis_in, label_array)
    y_pred = lr.predict_proba(Mahalanobis_in)[:, 1]
    print(lr.coef_)
    print(y_pred)
    for i in range(num_output):
        M_out = get_Mahalanobis_score_test(model, test_loader, 2, '/home/kiran/Desktop/Dev/SupCon/utils/output', True,
                                     'resnet', sample_mean, precision, i, .0014)
        M_out = np.asarray(M_out, dtype=np.float32)
        print(M_out.shape)
        if i == 0:
            Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
        else:
            Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)
    y_pred_test = lr.predict_proba(Mahalanobis_out)[:, 1]
    print(y_pred_test)
    for idx, (images, bcva, cst, eye_id, patient) in tqdm(enumerate(test_loader)):
        df.iloc[idx, 7] = Mahalanobis_out[idx,0] * lr.coef_[0][0] + Mahalanobis_out[idx,1] * lr.coef_[0][1] + Mahalanobis_out[idx,2] * lr.coef_[0][2] + Mahalanobis_out[idx,3] * lr.coef_[0][3] + Mahalanobis_out[idx,4] * lr.coef_[0][4]
    df.to_csv('prime_trex_ood_full_vit_mahal_correct.csv', index=False)
    '''
    for idx, (images, bcva, cst, eye_id, patient) in tqdm(enumerate(test_loader)):
        images = images.to(device)
        model.eval()
        features,out = model(images)
        #print(out)

        value = MSP(outputs=features,model = model)
        value = np.max(value)
        #print(value)
        value_odin = (testData(model,criterion,'cuda:0',.0014,1000,images))
        #print(value_odin)
        #value_mal_0 = get_Mahalanobis_score_ind(model, 2, 'resnet', sample_mean, precision, 0, .0014,images)
        #print(value_mal_0)
        #value_mal_1 = get_Mahalanobis_score_ind(model, 2, 'resnet', sample_mean, precision, 1, .0014, images)
        #value_mal_2 = get_Mahalanobis_score_ind(model, 2, 'resnet', sample_mean, precision, 2, .0014, images)
        #value_mal_3 = get_Mahalanobis_score_ind(model, 2, 'resnet', sample_mean, precision, 3, .0014, images)
        #value_mal_4 = get_Mahalanobis_score_ind(model, 2, 'resnet', sample_mean, precision, 4, .0014, images)

        mal_score = idx
        #print(mal_score)
        df.iloc[idx,5] = value
        df.iloc[idx,6] = value_odin
        df.iloc[idx,7] = mal_score
    df.to_csv('prime_trex_ood_full_vit.csv',index=False)
    '''
    #print(model)
    #get_Mahalanobis_score(model, test_loader, 2, 'resnet', sample_mean, precision,1, .0014)

def MSP(outputs, model):
    # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    return nnOutputs


def testData(net1, criterion, CUDA_DEVICE, noiseMagnitude1, temper,images):
    t0 = time.time()

    ###################################Out-of-Distributions#####################################


    inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad=True)
    outputs,out = net1(inputs)

    # Calculating the confidence of the output, no perturbation added here
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs[0]
    nnOutputs = nnOutputs - np.max(nnOutputs)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))


    # Using temperature scaling
    outputs = outputs / temper

    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    maxIndexTemp = np.argmax(nnOutputs)
    labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = (torch.ge(inputs.grad.data, 0))
    gradient = (gradient.float() - 0.5) * 2

    # Normalizing the gradient to the same space of image
    gradient[0][0] = (gradient[0][0]) #/ (63.0 / 255.0)
    #gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
    #gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)
    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
    outputs,out = net1(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs[0]
    nnOutputs = nnOutputs - np.max(nnOutputs)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))

    return np.max(nnOutputs)


def sample_estimator(model, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    for data, vit_deb,ir_hrf, full_vit,partial_vit,fluid_irf,drt,eye_id,bcva,cst,patient in tqdm(train_loader):
        total += data.size(0)
        data = data.cuda()
        target = full_vit
        data = Variable(data, volatile=True)
        output,out_features = model(data)
        #print("hello")
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
        #print(len(out_features))
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()

        # construct the sample matrix
        #print(data.size(0))
        #print(list_features)
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        # find inverse
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision

def get_Mahalanobis_score_test(model, test_loader, num_classes, outf, out_flag, net_type, sample_mean, precision,
                          layer_index, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []

    if out_flag == True:
        temp_file_name = '%s/confidence_Ga%s_In.txt' % (outf, str(layer_index))
    else:
        temp_file_name = '%s/confidence_Ga%s_Out.txt' % (outf, str(layer_index))

    g = open(temp_file_name, 'w')

    for data, bcva, cst, eye_id, patient in tqdm(test_loader):

        data = data.cuda()
        data = Variable(data, requires_grad=True)

        out_features = model(data)[1][layer_index]
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if net_type == 'densenet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
        elif net_type == 'resnet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
        tempInputs = torch.add(data.data, -magnitude, gradient)
        with torch.no_grad():
            noise_out_features = model(Variable(tempInputs))[1][layer_index]
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())

        for i in range(data.size(0)):
            g.write("{}\n".format(noise_gaussian_score[i]))
    g.close()

    return Mahalanobis
def get_Mahalanobis_score(model, test_loader, num_classes, outf, out_flag, net_type, sample_mean, precision,
                          layer_index, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []

    if out_flag == True:
        temp_file_name = '%s/confidence_Ga%s_In.txt' % (outf, str(layer_index))
    else:
        temp_file_name = '%s/confidence_Ga%s_Out.txt' % (outf, str(layer_index))

    g = open(temp_file_name, 'w')

    for data, vit_deb,ir_hrf, full_vit,partial_vit,fluid_irf,drt,eye_id,bcva,cst,patient in tqdm(test_loader):

        data = data.cuda()
        data = Variable(data, requires_grad=True)

        out_features = model(data)[1][layer_index]
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if net_type == 'densenet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
        elif net_type == 'resnet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
        tempInputs = torch.add(data.data, -magnitude, gradient)
        with torch.no_grad():
            noise_out_features = model(Variable(tempInputs))[1][layer_index]
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())

        for i in range(data.size(0)):
            g.write("{}\n".format(noise_gaussian_score[i]))
    g.close()

    return Mahalanobis
def get_Mahalanobis_score_ind(model, num_classes, net_type, sample_mean, precision,
                          layer_index, magnitude,data):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []





    #for data, bcva, cst, eye_id, patient in tqdm(test_loader):

    data = data.cuda()
    data= Variable(data, requires_grad=True)

    out_features = model(data)[1][layer_index]
    out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
    out_features = torch.mean(out_features, 2)
    #print(out_features)

    # compute Mahalanobis score
    gaussian_score = 0
    for i in range(0,num_classes):
        batch_sample_mean = sample_mean[layer_index][i]
        zero_f = out_features.data - batch_sample_mean
        term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        if i == 0:
            gaussian_score = term_gau.view(-1, 1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
    #print(gaussian_score)
    # Input_processing
    sample_pred = gaussian_score.max(1)[1]
    batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
    zero_f = out_features - Variable(batch_sample_mean)
    pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
    loss = torch.mean(-pure_gau)
    loss.backward()

    gradient = torch.ge(data.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    if net_type == 'densenet':
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                             gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(),gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(),gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
    elif net_type == 'resnet':
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                             gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
        #gradient.index_copy_(1, torch.LongTensor([1]).cuda(),gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
        #gradient.index_copy_(1, torch.LongTensor([2]).cuda(),gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
    tempInputs = torch.add(data.data, -magnitude, gradient)
    #print(tempInputs.shape)
    with torch.no_grad():
        noise_out_features = model(Variable(tempInputs))[1][layer_index]
    noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
    noise_out_features = torch.mean(noise_out_features, 2)
    #print(noise_out_features)
    noise_gaussian_score = 0
    for i in range(0,num_classes):
        batch_sample_mean = sample_mean[layer_index][i]
        zero_f = noise_out_features.data - batch_sample_mean
        term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        if i == 0:
            noise_gaussian_score = term_gau.view(-1, 1)
        else:
            noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

    noise_gaussian_score,_ = torch.max(noise_gaussian_score, dim=1)
    #print(noise_gaussian_score)
        #Mahalanobis.extend(noise_gaussian_score.cpu().numpy())



    return noise_gaussian_score.cpu().numpy()
def discretize(df):
    df = pd.read_csv(df)
    #df['MSP'] = pd.cut(df.MSP, bins=5000, labels=False)
    #df['ODIN'] = pd.cut(df.ODIN, bins=5000, labels=False)
    df['Distance'] = pd.cut(df.Distance, bins=10, labels=False)
    df.to_csv('/home/kiran/Desktop/Dev/FishEyeFord/contrastive_testing/patches_train_10.csv',index=False)
def discretize_labels(df_start, num_cuts):
    df = pd.read_csv(df_start)
    df['BCVA'] = pd.cut(df.BCVA, bins=num_cuts, labels=False)
    df['CST'] = pd.cut(df.CST, bins=num_cuts, labels=False)
    # df['Diabet'] = pd.qcut(df['Diabetes_Years'], q=num_cuts, labels=False,duplicates='drop')
    #df['DRSS'] = pd.qcut(df['DRSS'], q=num_cuts, labels=False, duplicates='drop')
    # df["Systolic BP"] = pd.qcut(df['Systolic BP'], q=num_cuts, labels=False)
    # df['Diastolic BP'] = pd.qcut(df['Diastolic BP'], q=num_cuts, labels=False)
    # df['HbAlC'] = pd.qcut(df['HbAlC'], q=num_cuts, labels=False)
    # file_name = './discretized_labels/' + 'recovery_' +str(num_cuts) +'.csv'
    # print(df['Diabetes_Years'].unique())
    file_name = '/home/kiran/Desktop/Dev/SupCon/final_csvs_1/Discretized_Datasets/cuts_' + str(num_cuts) + '.csv'
    df.to_csv(file_name, index=False)
    print(df.head())
if __name__ == '__main__':
    df = '/home/kiran/Desktop/Dev/FishEyeFord/contrastive_testing/patches_train.csv'
    discretize(df)
    #main()