import gzip
import pickle
import sys

def load_mnist(data_folder):
    f = gzip.open(data_folder + 'mnist.pkl.gz', 'rb')
    if sys.version_info < (3,):
        data = pickle.load(f)
    else:
        data = pickle.load(f, encoding='bytes')
    f.close()
    (x_train, y_train), (x_test, y_test) = data
    return x_train, y_train, x_test, y_test