from LwF_net import LwF

from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from dataset import CIFAR100

import numpy as np

from sklearn.model_selection import train_test_split

import math

import utils

import copy

import torch
from torch.autograd import Variable

####Hyper-parameters####
DEVICE = 'cuda'
BATCH_SIZE = 128
CLASSES_BATCH = 10
MEMORY_SIZE = 2000
########################

def main():

    range_classes = np.arange(100)
    classes_groups = np.array_split(range_classes, 10)
    
    perm_id = np.random.permutation(total_classes)
    all_classes = np.arange(total_classes)
    
    # Mix the classes indexes
    for i in range(len(all_classes)):
      all_classes[i] = perm_id[all_classes[i]]

    # Create class map
    class_map = {}
    #takes 10 new classes randomly
    for i, cl in enumerate(all_classes):
        class_map[cl] = i
    print (f"Class map:{class_map}\n")     
    
    # Create class map reversed
    map_reverse = {}
    for cl, map_cl in class_map.items():
        map_reverse[map_cl] = int(cl)
    print (f"Map Reverse:{map_reverse}\n")

    #path='orders/'
    #classes_groups, class_map, map_reverse = utils.get_class_maps_from_files(path+'classgroups1.pickle', path+'map1.pickle', path+'revmap1.pickle')
    #print(classes_groups, class_map, map_reverse)


    net = LwF(0, class_map)
    net.to(DEVICE)


    for i in range(int(100/CLASSES_BATCH)):
        
        print('-'*30)
        print(f'**** ITERATION {i+1} ****')
        print('-'*30)

        #torch.cuda.empty_cache()

        print('Loading the Datasets ...')
        print('-'*30)

        train_dataset, val_dataset, test_dataset = utils.get_datasets(classes_groups[i])

        print('-'*30)
        print('Updating representation ...')
        print('-'*30)

        net.update_representation(dataset=train_dataset, val_dataset=val_dataset, class_map=class_map, map_reverse=map_reverse)

        '''
        print('Reducing exemplar sets ...')
        print('-'*30)
        m = int(math.ceil(MEMORY_SIZE/net.n_classes))
        net.reduce_exemplars_set(m)
        print('Constructing exemplar sets ...')
        print('-'*30)
        for y in classes_groups[i]:
           net.construct_exemplars_set(train_dataset.dataset.get_class_imgs(y), m)
        '''

        net.n_known = net.n_classes

        print('Testing ...')
        print('-'*30)

        print('New classes')
        net.classify_all(test_dataset, map_reverse)

        if i > 0:

            previous_classes = np.array([])
            for j in range(i):
                previous_classes = np.concatenate((previous_classes, classes_groups[j]))

            prev_classes_dataset, all_classes_dataset = utils.get_additional_datasets(previous_classes, np.concatenate((previous_classes, classes_groups[i])))

            print('Old classes')
            net.classify_all(prev_classes_dataset, map_reverse)
            print('All classes')
            net.classify_all(all_classes_dataset, map_reverse)

            print('-'*30)

        #if i == 3:
            #return

if __name__ == '__main__':
    main()
