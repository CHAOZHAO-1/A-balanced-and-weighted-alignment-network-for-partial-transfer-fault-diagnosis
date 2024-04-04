import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torch
import itertools

import torch.nn as nn







class seCNN_1DmixAD(nn.Module):

    def __init__(self, num_classes=31):
        super(seCNN_1DmixAD, self).__init__()
        self.decay = 1
        self.n_class = num_classes
        # self.sharedNet = resnet18(False)

        self.sharedNet = CNN()
        self.cls_fc = nn.Linear(256, num_classes)
        self.domain_fc = AdversarialNetwork(in_feature=256)

        self.s_centroid = torch.zeros(self.n_class, 256)
        self.t_centroid = torch.zeros(self.n_class, 256)

        self.s_centroid = self.s_centroid.cuda()
        self.t_centroid = self.t_centroid.cuda()

        self.MSELoss = nn.MSELoss()  # (x-y)^2
        self.MSELoss = self.MSELoss.cuda()

    def forward(self, source,target):

        # source= source.unsqueeze(1)
        # target= target.unsqueeze(1)

        source_fea = self.sharedNet(source)

        target_fea = self.sharedNet(target)

        source=self.cls_fc(source_fea)
        target = self.cls_fc(target_fea)

        src_dlabel_pred = self.domain_fc(source_fea)
        tar_dlabel_pred = self.domain_fc(target_fea)



        return source,target,source_fea,target_fea,src_dlabel_pred,tar_dlabel_pred


    def adloss(self,s_feature, t_feature, y_s, y_t,type,ratio):

        weight=0
        n, d = s_feature.shape
        a=ratio


        # get labels
        s_labels, t_labels= y_s, torch.max(y_t, 1)[1]#得到源域和目标域标签



        t_labels=torch.cat((t_labels[:int(n*a)],s_labels[:int(n*(1-a))]), 0)

        t_feature=torch.cat((t_feature[:int(n*a)],s_feature[:int(n*(1-a))] ), 0)

        s_labels=s_labels[:int(n*(1-a))]
        s_feature=s_feature[:int(n*(1-a))]



        # image number in each class
        ones = torch.ones_like(s_labels, dtype=torch.float)
        ones_MIX = torch.ones_like(t_labels, dtype=torch.float)

        zeros = torch.zeros(self.n_class)

        zeros = zeros.cuda()
        s_n_classes = zeros.scatter_add(0, s_labels, ones)
        t_n_classes = zeros.scatter_add(0, t_labels, ones_MIX)




        # image number cannot be 0, when calculating centroids
        ones = torch.ones_like(s_n_classes)
        s_n_classes = torch.max(s_n_classes, ones)
        t_n_classes = torch.max(t_n_classes, ones)





        # calculating centroids, sum and divide
        zeros = torch.zeros(self.n_class, d)

        zeros = zeros.cuda()
        s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), s_feature)
        t_sum_feature = zeros.scatter_add(0, torch.transpose(t_labels.repeat(d, 1), 1, 0), t_feature)
        current_s_centroid = torch.div(s_sum_feature, s_n_classes.view(self.n_class, 1))
        current_t_centroid = torch.div(t_sum_feature, t_n_classes.view(self.n_class, 1))



        # Moving Centroid
        decay = self.decay
        s_centroid = (1-decay) * self.s_centroid + decay * current_s_centroid
        t_centroid = (1-decay) * self.t_centroid + decay * current_t_centroid


        #######select################





        if type=='SED':
            semantic_loss = self.MSELoss(s_centroid, t_centroid)
        if type=='WD':

            sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)
            sinkhorn.cuda()
            semantic_loss, P, C = sinkhorn(s_centroid,t_centroid)


        self.s_centroid = s_centroid.detach()
        self.t_centroid = t_centroid.detach()

        return semantic_loss


class CNN_1Dfea(nn.Module):

    def __init__(self, num_classes=31):
        super(CNN_1Dfea, self).__init__()

        self.sharedNet1 = CNN()
        self.sharedNet2 = CNN()

        self.cls_fc1 = nn.Linear(256, num_classes)
        self.cls_fc2 = nn.Linear(256, num_classes)


    def forward(self, source):

        source_fea = self.sharedNet1(source)
        source_lab =self.cls_fc1(source_fea)
        source_lab=torch.max(source_lab, 1)[1]

        return source_fea,source_lab




class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, num_classes=10):
        super(CNN, self).__init__()


        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64,stride=1),  # 32, 24, 24
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )  # 32, 12,12     (24-2) /2 +1


        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16,stride=1),  # 128,8,8
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))# 128, 4,4

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5,stride=1),  # 32, 24, 24
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 32, 12,12     (24-2) /2 +1

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5,stride=1),  # 128,8,8
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4)
        )





        # self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        x = x.unsqueeze(1)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)

        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)


        x = x.view(x.size(0), -1)

        # x = self.layer5(x)

        # x = self.fc(x)

        return x





class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 128)
        self.ad_layer2 = nn.Linear(128, 2)
        # self.ad_layer1.weight.data.normal_(0, 0.01)
        # self.ad_layer2.weight.data.normal_(0, 0.3)
        # self.ad_layer1.bias.data.fill_(0.0)
        # self.ad_layer2.bias.data.fill_(0.0)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        # print(x.size())
        # x = self.relu2(x)
        # x = self.dropout2(x)
        # x = self.sigmoid(x)
        return x

    def output_num(self):
        return 1






