#author:zhaochao time:2020/7/27
import pickle
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import data_loader_1d
import resnet18_1d as models
from torch.utils import model_zoo
import  time
import numpy as np
import  random
import torch.nn as nn

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速


def EntropyLoss(input_):
    mask = input_.ge(0.000001)# 逐元素比较
    mask_out = torch.masked_select(input_, mask)# 筛选
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))




os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training settings


momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4



def train(model):
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_train_loader)


    src_dlabel = Variable(torch.ones(batch_size).long().cuda())
    tgt_dlabel = Variable(torch.zeros(batch_size).long().cuda())


    Train_Loss_list = []
    Train_Accuracy_list = []
    Test_Loss_list = []
    Test_Accuracy_list = []

    # class_weight_list = np.zeros((1, class_num))
    correct = 0
    start=time.time()
    for i in range(1, iteration + 1):
        model.train()
        LEARNING_RATE = Glr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        D_LEARNING_RATE = Dlr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print('learning rate{: .4f}'.format(LEARNING_RATE))

        optimizer = torch.optim.Adam([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10,weight_decay=l2_decay)

        optimizer_critic = torch.optim.Adam([
            {'params': model.domain_fc.parameters()},
        ], lr=D_LEARNING_RATE, weight_decay=l2_decay)



        try:
            src_data, src_label = src_iter.next()

            tgt_data, _ = tgt_iter.next()
        except Exception as err:
            src_iter = iter(src_loader)
            src_data, src_label = src_iter.next()
            tgt_iter = iter(tgt_train_loader)
            tgt_data, _ = tgt_iter.next()

        if i % tgt_loader_len == 0:
            tgt_iter = iter(tgt_train_loader)
        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()

            tgt_data = tgt_data.cuda()

        optimizer.zero_grad()

        src_pred,tar_pred,source_fea,target_fea,src_dlabel_pred,tgt_dlabel_pred= model(src_data, tgt_data)



        semantic_loss= model.adloss(source_fea, target_fea, src_label, tar_pred, distype,ratio)


####################
        new_label_pred = torch.cat((src_dlabel_pred, tgt_dlabel_pred), 0)
        confusion_loss = nn.CrossEntropyLoss(reduction='none')
        confusion_loss_total = confusion_loss(new_label_pred, torch.cat((src_dlabel, tgt_dlabel), 0))



##################
        m = nn.Softmax(dim=1)
        class_weight = torch.sum(m(tar_pred), dim=0) / batch_size
        class_weight = class_weight / max(class_weight)


############
        t_labels = torch.max(tar_pred, 1)[1]

        class_label = torch.cat((src_label, t_labels), 0)
        s_one_hot = torch.zeros(2 * batch_size, class_num).cuda().scatter_(1, class_label.reshape(-1, 1), 1)
        weight = s_one_hot.mm(class_weight.reshape(-1, 1).cuda())
        weight[batch_size:] = 1

        confusion_loss_total = torch.sum((confusion_loss_total.reshape(-1, 1)).mul(weight)) / (2 * batch_size)

############
        H = EntropyLoss(m(tar_pred))

        loss = nn.CrossEntropyLoss()
        label_loss = loss(src_pred, src_label)





        lambd = 2 / (1 + math.exp(-10 * (i) / iteration)) - 1

        loss = label_loss + semantic_loss*c-a*lambd*confusion_loss_total+H*lambd*h

        loss.backward(retain_graph=True)
        optimizer.step()

        optimizer_critic.zero_grad()
        confusion_loss_total.backward(retain_graph=True)
        optimizer_critic.step()




        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tsemantic_Loss: {:.6f}\tcon_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), label_loss, semantic_loss,confusion_loss_total))

        if i % (log_interval * 10) == 0:

            train_correct,train_loss = test(model,src_loader)
            test_correct, test_loss = test(model,tgt_test_loader)


            print('src: {} to tgt: {} max correct: {} max accuracy{: .2f}%\n'.format(
                src_name, tgt_name, correct, 10000. * correct / tgt_dataset_len))

            Train_Accuracy_list.append(train_correct.cpu().numpy() / len(src_loader.dataset))
            Train_Loss_list.append(train_loss)

            Test_Accuracy_list.append(test_correct.cpu().numpy() / len(tgt_test_loader.dataset))
            Test_Loss_list.append(test_loss)



def test(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            tgt_pred,_,_,_,_,_ = model(tgt_test_data, tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

    test_loss /= tgt_dataset_len
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        tgt_name, test_loss, correct, len(test_loader.dataset),
        10000. * correct / len(test_loader.dataset)))
    return correct,test_loss


if __name__ == '__main__':

    distype = 'SED'

    iteration = 10000

    FFT = True

    ratiolist=[30/32]

    dataset = 'PU2'

    class_num = 12

    c=0.2

    a=0.5

    h=0.1

    Task_name = np.array(['C1', 'C2', 'C3'])

    src_tar = np.array([[6, 9], [6, 9], [6, 9]])

    Target_class = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 3, 4, 6, 7, 8, 9, 10, 11],
        [2, 6, 7]
    ])

    for TT in range(3):
        print(Task_name[TT])

        batch_size = 512
        all_category = np.linspace(1, class_num, class_num, endpoint=True)
        partial = Target_class[TT]
        notargetclasses = list(set(all_category) ^ set(partial))
        Glr = 0.01
        Dlr = 0.001

        source = src_tar[TT][0]
        target = src_tar[TT][1]

        for repeat in range(10):

            root_path = '/home/dlzhaochao/deeplearning/DTL/data/' + dataset + 'data' + str(class_num) + '.mat'
            src_name = 'load' + str(source) + '_train'
            tgt_name = 'load' + str(target) + '_train'
            test_name = 'load' + str(target) + '_test'

            cuda = not no_cuda and torch.cuda.is_available()
            torch.manual_seed(seed)
            if cuda:
                torch.cuda.manual_seed(seed)

            kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

            src_loader = data_loader_1d.load_training([], root_path, src_name, FFT, class_num, batch_size,
                                                      kwargs)
            tgt_train_loader = data_loader_1d.load_training(notargetclasses, root_path, tgt_name, FFT,
                                                            class_num, batch_size, kwargs)

            tgt_test_loader = data_loader_1d.load_testing(notargetclasses, root_path, test_name, FFT,
                                                          class_num,
                                                          batch_size, kwargs)

            src_dataset_len = len(src_loader.dataset)
            tgt_dataset_len = len(tgt_test_loader.dataset)
            src_loader_len = len(src_loader)
            tgt_loader_len = len(tgt_train_loader)

            model = models.seCNN_1DmixAD(num_classes=class_num)

            print(model)
            if cuda:
                model.cuda()
            train(model)





































