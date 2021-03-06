import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from resnet import resnet32

import copy

import math

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

####Hyper-parameters####
LR = 2
WEIGHT_DECAY = 0.00001
BATCH_SIZE = 128
STEPDOWN_EPOCHS = [49, 63]
STEPDOWN_FACTOR = 5
NUM_EPOCHS = 2
DEVICE = 'cuda'
########################

def validate(net, val_dataloader, map_reverse):
    running_corrects_val = 0
    for inputs, labels, index in val_dataloader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        net.train(False)
        # forward
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
        running_corrects_val += (preds == labels.cpu().numpy()).sum()
        #running_corrects_val += torch.sum(preds == labels.data)

    valid_acc = running_corrects_val / float(len(val_dataloader.dataset))

    net.train(True)
    return valid_acc


class LwF(nn.Module):
    def __init__(self, n_classes, class_map):
        super(LwF, self).__init__()
        self.features_extractor = resnet32(num_classes=0)

        self.n_classes = n_classes
        self.n_known = 0

        self.clf_loss = nn.BCEWithLogitsLoss()
        self.dist_loss = nn.BCEWithLogitsLoss()

        self.class_map = class_map


    def forward(self, x):
        x = self.features_extractor(x)
        return x

    def add_classes(self, n):
        in_features = self.features_extractor.fc.in_features
        out_features = self.features_extractor.fc.out_features
        weight = copy.deepcopy(self.features_extractor.fc.weight.data)
        bias = copy.deepcopy(self.features_extractor.fc.bias.data)

        self.features_extractor.fc = nn.Linear(in_features, out_features+n)
        self.features_extractor.fc.weight.data[:out_features] = copy.deepcopy(weight)
        self.features_extractor.fc.bias.data[:out_features] = copy.deepcopy(bias)

        self.n_classes += n

    def update_representation(self, dataset, val_dataset, class_map, map_reverse):
        dataset = dataset.dataset
        targets = list(set(dataset.targets))
        n = len(targets)

        print('New classes:{}'.format(n))
        print('-'*30)
        
            
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
 
        self.features_extractor.to(DEVICE)
        if self.n_known > 0:
            #self.features_extractor.to(DEVICE)
            #self.features_extractor.train(False)
            q = torch.zeros(len(dataset), self.n_classes).cuda()
            q_val = torch.zeros(len(val_dataset), self.n_classes).cuda()
            
            for images, labels, indexes in loader:
                images = Variable(images).cuda()
                indexes = indexes.cuda()
                self.features_extractor.train(False)
                g = torch.sigmoid(self.features_extractor.forward(images))
                #g = self.forward(images)
                q[indexes] = g.data
            q = Variable(q).cuda()
            
         
            for images_v,labels_v,indexes_v in val_loader:
                images_v = Variable(images_v).cuda()
                indexes_v = indexes_v.cuda()
                g_val = torch.sigmoid(self.features_extractor.forward(images_v))
                #g = self.forward(images)
                q_val[indexes_v] = g_val.data
            q_val = Variable(q_val).cuda()
            
            self.features_extractor.train(True)
            
        self.add_classes(n)
        #self.n_classes += n

        optimizer = optim.SGD(self.parameters(), lr=2.0, weight_decay=0.00001)

        i = 0

        #best_acc = -1
        best_epoch = 0
        val_loss = 0.0
        min_val_loss = None
        
        self.to(DEVICE)
        self.features_extractor.train(True)
        for epoch in range(NUM_EPOCHS):
            
            val_loss = 0.0
            min_val_loss = None
            
            if epoch in STEPDOWN_EPOCHS:
         
              for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/STEPDOWN_FACTOR


            for imgs, labels, indexes in loader:
                imgs = imgs.to(DEVICE)
                indexes = indexes.to(DEVICE)
                # We need to save labels in this way because classes are randomly shuffled at the beginning
                seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
                labels = Variable(seen_labels).to(DEVICE)
                labels_hot=torch.eye(self.n_classes)[labels]
                labels_hot = labels_hot.to(DEVICE)
                
                self.features_extractor.train(True)

                optimizer.zero_grad()
                #out = torch.sigmoid(self(imgs))
                out = self(imgs)

                #print(out[0])

                #print('out', out[0], 'labels', labels[0])

                if self.n_known <= 0:
                    loss = self.clf_loss(out, labels_hot)

                else:
                    
                    #out = torch.sigmoid(out)
                    q_i = q[indexes]
                    target = torch.cat((q_i[:, :self.n_known], labels_hot[:, self.n_known:self.n_classes]), dim=1)
                    loss = self.dist_loss(out, target)
                    
                    
                    #loss += dist_loss

                loss.backward()
                optimizer.step()
                self.features_extractor.train(False)
                self.train(False)
                val_loss = 0.0
                for inputs_v, labels_v, indexes_v in val_loader:
                   
                      
                       inputs_v, indexes_v = inputs_v.to(DEVICE), indexes_v.to(DEVICE)
                       lebels_v = labels_v.to(DEVICE)
                       seen_labels_v = torch.LongTensor([class_map[label] for label in labels_v.cpu().numpy()])
                       labels_v = Variable(seen_labels_v).to(DEVICE)
                       labels_hot_v =torch.eye(self.n_classes)[labels_v]
                       labels_hot_v = labels_hot_v.to(DEVICE)
                       out_v = self(inputs_v)
                        
                       if self.n_known <= 0:
                            
                                val_loss = self.clf_loss(out_v, labels_hot_v)* inputs_v.size(0)
                       else:
                                q_val_i = q_val[indexes_v]
                                target_v = torch.cat((q_val_i[:, :self.n_known], labels_hot_v[:, self.n_known:self.n_classes]), dim=1)
                                val_loss += self.dist_loss(out_v, target_v).item()* inputs_v.size(0)
                                
                self.features_extractor.train(True)                
                self.train(True) 
                                
                avg_val_loss = val_loss / float(len(val_loader.dataset))
            
                        
            ''' accuracy = validate(self, val_loader, map_reverse)

            if accuracy > best_acc:
                best_acc = accuracy
                best_epoch = epoch
                best_net = copy.deepcopy(self.state_dict()) '''
            
            if min_val_loss is None:
                min_val_loss = val_loss
                best_net = copy.deepcopy(self.state_dict())
            else:
                if val_loss < min_val_loss:
                    best_epoch = epoch
                    min_val_loss = val_loss
                    best_net = copy.deepcopy(self.state_dict())

            if i % 10 == 0 or i == (NUM_EPOCHS-1):
                print('Epoch {} Loss:{:.4f}'.format(i, loss.item()))
                for param_group in optimizer.param_groups:
                  print('Learning rate:{}'.format(param_group['lr']))
                #print('Max Accuracy:{:.4f} (Epoch {})'.format(best_acc, best_epoch))
                print('Min Validation loss: {:.4f} (Epoch {})'.format(min_val_loss,best_epoch))
                print('-'*30)
            i+=1

        self.load_state_dict(best_net)
        return



    def classify_all(self, test_dataset, map_reverse):

        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        running_corrects = 0
        #self.features_extractor(DEVICE)
        #self.features_extractor.train(False)

        for imgs, labels, _ in  test_dataloader:
            imgs = Variable(imgs).cuda()
            self.features_extractor.train(False)
            _, preds = torch.max(self(imgs), dim=1)
            preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
            running_corrects += (preds == labels.numpy()).sum()
            #running_corrects += torch.sum(preds == labels.data).data.item()
        self.features_extractor.train(True)
        accuracy = running_corrects / float(len(test_dataloader.dataset))
        print('Test Accuracy: {}'.format(accuracy))
