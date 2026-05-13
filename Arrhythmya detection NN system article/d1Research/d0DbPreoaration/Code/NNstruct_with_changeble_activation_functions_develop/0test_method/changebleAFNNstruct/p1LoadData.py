import numpy as np
import torchvision
from sklearn.model_selection import train_test_split

def showTrainTestSizes(X_train, X_test, y_train, y_test):
    print('-------------------------------------')
    print(f'X_train size: {X_train.size()}')
    print(f'X_test size: {X_test.size()}')
    print(f'y_train size: {y_train.size()}')
    print(f'y_test size: {y_test.size()}')
    print('-------------------------------------')

def loadMNIST():
    # Load MNIST data
    data = torchvision.datasets.MNIST('./data', download=True)
    print(f'input datashape {np.shape(data.data)}')

    # reshape data
    X_data_reshaped = data.data.reshape(-1, 784).float() / 255.0
    Y_data_reshaped = data.targets
    print(f'reshaped datashape X {np.shape(X_data_reshaped)}')
    print(f'reshaped datashape Y {np.shape(Y_data_reshaped)}')

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data_reshaped, Y_data_reshaped, test_size=0.1, random_state=42)
    print(f'Train size X:{np.shape(X_train)}, Y:{np.shape(y_train)}')
    print(f'Test size X:{np.shape(X_test)}, Y:{np.shape(y_test)}')
    return X_train, X_test, y_train, y_test

