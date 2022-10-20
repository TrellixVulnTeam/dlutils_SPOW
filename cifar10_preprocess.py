import tarfile
import pickle
import random
import numpy as np
import tensorflow as tf
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm

from utils.data_preprocess import normalize, integer_to_one_hot

class DownloadProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

"""
    check if the data (zip) file is already downloaded
    if not, download it from "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" and save as cifar-10-python.tar.gz
"""

def load_preprocessed_cifar10(cifar10_dataset_root_folder_path):
    x_train = np.load(cifar10_dataset_root_folder_path+'x_train.npy')
    y_train = np.load(cifar10_dataset_root_folder_path+'y_train.npy')
    x_test = np.load(cifar10_dataset_root_folder_path+'x_test.npy')
    y_test = np.load(cifar10_dataset_root_folder_path+'y_test.npy')
    return x_train, y_train, x_test, y_test

def load_cifar10(cifar10_dataset_root_folder_path):
    # Download the files if not.
    get_batches(cifar10_dataset_root_folder_path)
    # Preprocess all the data and save it
    preprocess_and_save_data(cifar10_dataset_root_folder_path)
    # load the saved dataset
    train_data_1, train_labels_1 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_batch_1.p', mode='rb'))
    train_data_2, train_labels_2 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_batch_2.p', mode='rb'))
    train_data_3, train_labels_3 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_batch_3.p', mode='rb'))
    train_data_4, train_labels_4 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_batch_4.p', mode='rb'))
    train_data_5, train_labels_5 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_batch_5.p', mode='rb'))
    train_data_6, train_labels_6 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_validation.p', mode='rb'))
    train_data = np.concatenate((train_data_1, train_data_2, train_data_3, train_data_4 , train_data_5, train_data_6))
    train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3, train_labels_4, train_labels_5, train_labels_6))
    test_data, test_labels = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_test.p', mode='rb'))
    return train_data, train_labels, test_data, test_labels

def load_cifar10_with_batches(cifar10_dataset_root_folder_path):
    # Download the files if not.
    get_batches(cifar10_dataset_root_folder_path)
    # Preprocess all the data and save it
    preprocess_and_save_data(cifar10_dataset_root_folder_path)
    # load the saved dataset
    train_data_1, train_labels_1 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_batch_1.p', mode='rb'))
    train_data_2, train_labels_2 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_batch_2.p', mode='rb'))
    train_data_3, train_labels_3 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_batch_3.p', mode='rb'))
    train_data_4, train_labels_4 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_batch_4.p', mode='rb'))
    train_data_5, train_labels_5 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_batch_5.p', mode='rb'))
    train_data_6, train_labels_6 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_validation.p', mode='rb'))
    test_data, test_labels = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_test.p', mode='rb'))
    return train_data_1, train_labels_1, train_data_2, train_labels_2, train_data_3, train_labels_3, train_data_4, train_labels_4, train_data_5, train_labels_5, train_data_6, train_labels_6, test_data, test_labels

def load_cifar10_with_validation(cifar10_dataset_root_folder_path):
    # Download the files if not.
    get_batches(cifar10_dataset_root_folder_path)
    # Preprocess all the data and save it
    preprocess_and_save_data(cifar10_dataset_root_folder_path)
    # load the saved dataset
    train_data_1, train_labels_1 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_batch_1.p', mode='rb'))
    train_data_2, train_labels_2 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_batch_2.p', mode='rb'))
    train_data_3, train_labels_3 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_batch_3.p', mode='rb'))
    train_data_4, train_labels_4 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_batch_4.p', mode='rb'))
    train_data_5, train_labels_5 = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_batch_5.p', mode='rb'))
    train_data = np.concatenate((train_data_1, train_data_2, train_data_3, train_data_4 , train_data_5))
    train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3, train_labels_4, train_labels_5))
    val_data, val_labels = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_validation.p', mode='rb'))
    test_data, test_labels = pickle.load(open(cifar10_dataset_root_folder_path + 'preprocess_test.p', mode='rb'))
    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def get_batches(cifar10_dataset_root_folder_path):
    # Download the dataset (if not exist yet)
    if not isfile(cifar10_dataset_root_folder_path + 'cifar-10-python.tar.gz'):
        with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
            urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', 'cifar-10-python.tar.gz', pbar.hook)

    if not isdir(cifar10_dataset_root_folder_path + 'cifar-10-batches-py'):
        with tarfile.open('cifar-10-python.tar.gz') as tar:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar)
            tar.close()

def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

def _preprocess_and_save(features, labels, filename):
    features = normalize(features)
    labels = integer_to_one_hot(labels)

    pickle.dump((features, labels), open(filename, 'wb'))

def preprocess_and_save_data(cifar10_dataset_root_folder_path):
    n_batches = 5
    valid_features = []
    valid_labels = []
    cifar10_dataset_folder_path = cifar10_dataset_root_folder_path + 'cifar-10-batches-py'

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)

        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)

        # preprocess the 90% of the whole dataset of the batch
        # - normalize the features
        # - one_hot_encode the lables
        # - save in a new file named, "preprocess_batch_" + batch_number
        # - each file for each batch
        if not isfile(cifar10_dataset_root_folder_path + 'preprocess_batch_' + str(batch_i) + '.p'):
            _preprocess_and_save(features[:-index_of_validation], labels[:-index_of_validation],
                                cifar10_dataset_root_folder_path + 'preprocess_batch_' + str(batch_i) + '.p')

        # unlike the training dataset, validation dataset will be added through all batch dataset
        # - take 10% of the whold dataset of the batch
        # - add them into a list of
        #   - valid_features
        #   - valid_labels
        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])

    # preprocess the all stacked validation dataset
    if not isfile(cifar10_dataset_root_folder_path + 'preprocess_validation.p'):
        _preprocess_and_save(np.array(valid_features), np.array(valid_labels),
                            cifar10_dataset_root_folder_path + 'preprocess_validation.p')

    if not isfile(cifar10_dataset_root_folder_path + 'preprocess_test.p'):
        # load the test dataset
        with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')

        # preprocess the testing data
        test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        test_labels = batch['labels']

        # Preprocess and Save all testing data
        _preprocess_and_save(np.array(test_features), np.array(test_labels),
                            cifar10_dataset_root_folder_path + 'preprocess_test.p')