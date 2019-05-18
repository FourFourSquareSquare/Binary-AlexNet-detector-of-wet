"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
Modified by: Travis Shao
contact: f.kratzert(at)gmail.com
"""

import os

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
Iterator = tf.data.Iterator

"""
Configuration Part.
"""

# Path to the textfiles for the trainings and validation set
TRAIN_FILE = '.../train.txt'
VAL_FILE = '.../val.txt'

# Learning params
LEARNING_RATE = 0.00005
NUM_EPOCHS = 20
BATCH_SIZE = 16

# Network params
DROPOUT_RATE = 0.5
NUM_CLASSES = 2
TRAIN_LAYERS = ['fc8']

# How often we want to write the tf.summary data to disk
DISPLAY_STEP = 1

# Path for tf.summary.FileWriter and to store model checkpoints
FILEWRITER_PATH = ".../tensorboard"
CHECKPOINT_PATH = ".../checkpoints"

# For accuracy.txt
ACCURACY_PATH = '.../accuracy.txt'
ACCURACY_OUTPUT = open(ACCURACY_PATH, 'w')

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(CHECKPOINT_PATH):
    os.mkdir(CHECKPOINT_PATH)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(TRAIN_FILE,
                                 mode='training',
                                 batch_size=BATCH_SIZE,
                                 num_classes=NUM_CLASSES,
                                 shuffle=True)
    val_data = ImageDataGenerator(VAL_FILE,
                                  mode='inference',
                                  batch_size=BATCH_SIZE,
                                  num_classes=NUM_CLASSES,
                                  shuffle=False)
                                  
    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [BATCH_SIZE, 227, 227, 3])
y = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, NUM_CLASSES, TRAIN_LAYERS)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in TRAIN_LAYERS]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

saver = tf.train.Saver()

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(FILEWRITER_PATH)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/BATCH_SIZE))
val_batches_per_epoch = int(np.floor(val_data.data_size / BATCH_SIZE))

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)
    
    # If loading from checkpoints
    # saver = tf.train.import_meta_graph('.../checkpoints/model_epoch20.ckpt.meta')
    # saver.restore(sess, ".../checkpoints/model_epoch20.ckpt")
    
    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                        FILEWRITER_PATH))

    print("{} Start validation".format(datetime.now()))
    sess.run(validation_init_op)
    test_acc = 0.
    test_count = 0
    for _ in range(val_batches_per_epoch):

        img_batch, label_batch = sess.run(next_batch)
        acc = sess.run(accuracy, feed_dict={x: img_batch,
                                            y: label_batch,
                                            keep_prob: 1.})
        test_acc += acc
        test_count += 1
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
    # Loop over number of epochs
    for epoch in range(NUM_EPOCHS):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            # And run the training op
            sess.run(train_op, feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: DROPOUT_RATE})
            print("{} Step = ", step)
            # Generate summary with the current batch of data and write to file
            if step % DISPLAY_STEP == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch*train_batches_per_epoch + step)
                
        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # Write to accuracy.txt for data recording purposes
        ACCURACY_OUTPUT.write(str(test_acc))
        ACCURACY_OUTPUT.write("\n")

        # save checkpoint of the model
        checkpoint_name = os.path.join(CHECKPOINT_PATH,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
