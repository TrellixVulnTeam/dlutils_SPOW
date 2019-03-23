import numpy as np
from utils.data_preprocess import *

def display_stats(data, labels, label_names):

    print('\nStats of batch:')
    print('# of Samples: {}\n'.format(len(data)))

    if is_one_hot(labels):
        labels_integer = one_hot_to_integer(labels)

    label_counts = dict(zip(*np.unique(labels_integer, return_counts=True)))
    for key, value in label_counts.items():
        print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))

    print('Image - Min Value: {} Max Value: {}'.format(data.min(), data.max()))
    print('Image - Shape: {}'.format(data[0].shape))

def display_data_props(data, tag="Data"):
    print(tag + " properties:")
    print("Shape: {}".format(data.shape))
    print("Type: {}".format(type(data)))
    #print("Head5:"), print(train_data.head(5))
    #print(train_data.isnull().any().describe())
    print("")

def display_train_test_props(train_data, test_data):
    display_data_props(train_data, "Training data")
    display_data_props(test_data, "Test data")