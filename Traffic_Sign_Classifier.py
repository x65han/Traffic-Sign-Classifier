# Load pickled data
import pickle

data_directory = "./data_sets/"
training_file = data_directory + "train.p"
validation_file = data_directory + "valid.p"
testing_file = data_directory + "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# print data shapes
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_valid:", X_valid.shape)
print("y_valid:", y_valid.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
print("Sample input shape:", X_train[0].shape)
print("Sample output:", y_train[0])

# Dataset Summary & Exploration
import numpy as np
# Number of training examples
n_train = len(X_train)
# Number of testing examples
n_test = len(X_test)
# shape of input image
image_shape = X_train[0].shape
# Number of unique labels
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Shape of image data =", image_shape)
print("Number of classes =", n_classes)

# Visualize data
import matplotlib.pyplot as plt
import random

def visualize_data():
    fig, axs = plt.subplots(3, 5, figsize=(15, 6))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()

    for i in range(15):
        index = random.randint(0, len(X_train) - 1)
        image = X_train[index]
        axs[i].axis('off')
        axs[i].imshow(image)
        axs[i].set_title(y_train[index])
    plt.show()

visualize_data()


