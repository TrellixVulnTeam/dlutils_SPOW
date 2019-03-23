import numpy as np
from sklearn.model_selection import train_test_split

def is_one_hot(labels):
    """Returns true if the given labels are one-hot encoded. 
    Returns False otherwise"""
    return (labels.max() == 1 and labels.min() == 0)

def integer_to_one_hot(y_train, y_test=None):
    """One hot encodes the labels"""
    # +1 because we have 0 as label too
    encoded = np.zeros((len(y_train), y_train.max()+1))
    for idx, val in enumerate(y_train):
        encoded[idx][val] = 1
    
    if y_test is not None:
        encoded2 = np.zeros((len(y_test), y_test.max()+1))
        for idx, val in enumerate(y_test):
            encoded2[idx][val] = 1
        return encoded, encoded2
    else:
        return encoded

def one_hot_to_integer(labels):
    """Converts one-hot encoded labels to integer encoded"""
    return np.asarray([np.where(r==1)[0][0] for r in labels])

def normalize(x_train, x_test=None):
    """ Normalize the data between 0.0 - 1.0"""
    min_val = np.min(x_train)
    max_val = np.max(x_train)
    x_train = (x_train-min_val) / (max_val-min_val)
    if x_test is not None:
        min_val = np.min(x_test)
        max_val = np.max(x_test)
        x_test = (x_test-min_val) / (max_val-min_val)
        return x_train, x_test
    else:
        return x_train

def split_data(data, labels, test_size, random_seed=0):
    """Splits data according to the given ration of test_size(0.0-1.0)"""
    train_test_split(data, labels, test_size=test_size, random_seed=random_seed)