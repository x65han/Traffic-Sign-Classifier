# Load pickled data
import pickle

data_directory = "./data_sets/"
with open(data_directory + "train.p", mode='rb') as f:
    train = pickle.load(f)
with open(data_directory + "test.p", mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test,  y_test  = test['features'],  test['labels']

# Build Validation Set
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

# Dataset Summary & Exploration
assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))
import numpy as np
# Number of training examples
n_train = len(X_train)
# Number of testing examples
n_test = len(X_test)
# Number of validation examples
n_valid = len(X_valid)
# shape of input image
image_shape = X_train[0].shape
# Number of unique labels
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Shape of image data =", image_shape)
print("Number of classes =", n_classes)

# Visualize data
import matplotlib.pyplot as plt
import random

def visualize_data(X_data, y_data, title="No Title", gray_scale=False):
    fig, axs = plt.subplots(3, 5, figsize=(15, 6))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()

    for i in range(15):
        index = random.randint(0, len(X_data) - 1)
        image = X_data[index]
        axs[i].axis('off')
        if gray_scale == True:
            axs[i].imshow(image.squeeze(), cmap='gray')
        else:
            axs[i].imshow(image)
        axs[i].set_title(y_data[index])
    fig.canvas.set_window_title(title)
    plt.show()

# visualize_data(X_train, y_train, title="Training Sample")

# Pre-process Data
from sklearn.utils import shuffle

# Pre-process Data: shuffle data
X_train, y_train = shuffle(X_train, y_train)

# Pre-process Data: Normalization
# X_train = (X_train - 128) / 128
# X_test = (X_test - 128)  / 128

# Pre-process Data: Grayscale
X_train = np.sum(X_train / 3, axis=3, keepdims=True)
X_test  = np.sum(X_test  / 3, axis=3, keepdims=True)
X_valid = np.sum(X_valid / 3, axis=3, keepdims=True)

# visualize_data(X_train, y_train, title="Pre-processed Sample", gray_scale=True)

# Setup Tensorflow
import tensorflow as tf
from tensorflow.contrib.layers import flatten
EPOCHS = 20
rate = 0.001
keep_prob = 1
BATCH_SIZE = 128

# CNN
def LeNet(x):
    # hyper-parameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Layer 1: Activation.
    conv1 = tf.nn.relu(conv1)

    # Layer 1: Dropout
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # Layer 1: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Layer 2: Activation.
    conv2 = tf.nn.relu(conv2)

    # Layer 2: Dropout
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # Layer 2: Pooling. Input = 10x10x16. Output = 5x5x16
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten Neural Network. Input = 5x5x16. Output = 400.
    conv2 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = n_classes.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(400, n_classes), mean=mu, stddev=sigma))
    conv3_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(conv2, conv3_W) + conv3_b

    return logits

# Features and Labels
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

# Training Pipeline
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

# Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_calculation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_calculation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Train the Model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {}/{}:".format(i + 1, EPOCHS))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    saver.save(sess, './tf')
    print("Model saved")

# Test the Model
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
