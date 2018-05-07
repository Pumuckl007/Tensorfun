import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import os
import random
import math
print ("PACKAGES LOADED")

files = os.listdir("/media/pics/chars/")
files = sorted(files)

training = []
testing = []

for file in files:
    img = cv2.imread("/media/pics/chars/" + file, cv2.IMREAD_GRAYSCALE)
    # img = cv2.inRange(img, 40, 255)
    np_image_data = np.asarray(img)
    np_image_data=cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
    np_final = np.concatenate(np_image_data,axis=0)
    hot = ord(file[0].upper())-65
    label = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    label[hot] = 1
    if random.randint(0, 100) < 2:
        testing.append({'img': np_final, 'label': label})
    else:
        training.append({'img': np_final, 'label': label})

random.shuffle(training)


x = tf.placeholder("float", [None, 100])
y = tf.placeholder("float", [None, 26])  # None is for infinite
W = tf.Variable(tf.zeros([100, 26]))
b = tf.Variable(tf.zeros([26]))
# LOGISTIC REGRESSION MODEL
actv = tf.nn.softmax(tf.matmul(x, W) + b)
# COST FUNCTION
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv), axis=1))
# OPTIMIZER
learning_rate = 0.5
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# PREDICTION
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
# ACCURACY
accr = tf.reduce_mean(tf.cast(pred, "float"))
# INITIALIZER
init = tf.global_variables_initializer()

batch_size      = 400
training_epochs = int(math.floor(len(training)/400))
display_step    = 5
# SESSION
sess = tf.InteractiveSession()
sess.run(init)

def getBatch(start, num):
    data = training[start:(start+num)]
    return data

def getXsAndYs(batch):
    xs = []
    ys = []
    for data in batch:
        xs.append(data['img'])
        ys.append(data['label'])
    return xs, ys

current_pos = 0
# MINI-BATCH LEARNING
for epoch in range(training_epochs):
    avg_cost = 0.
    num_batch = int(len(training)/batch_size)
    for i in range(num_batch):
        batch = getBatch(current_pos, batch_size)
        batch_xs, batch_ys = getXsAndYs(batch)
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
        feeds = {x: batch_xs, y: batch_ys}
        avg_cost += sess.run(cost, feed_dict=feeds)/num_batch
    # DISPLAY
    if epoch % display_step == 0:
        feeds_train = {x: batch_xs, y: batch_ys}
        test_xs, test_ys = getXsAndYs(testing)
        feeds_test = {x: test_xs, y: test_ys}
        train_acc = sess.run(accr, feed_dict=feeds_train)
        test_acc = sess.run(accr, feed_dict=feeds_test)
        print ("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f"
               % (epoch, training_epochs, avg_cost, train_acc, test_acc))
    current_pos += batch_size
print ("DONE")

saver = tf.train.Saver()
saver.save(sess, "/media/pics/model.ckpt")
