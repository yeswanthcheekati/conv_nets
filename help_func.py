import tensorflow as tf
import numpy as np
from PIL import Image
from scipy import ndimage
import random
import skfmm


def get_labels(file_path):
    sims = []
    sim_u = []
    for i in range(25):
        file = Image.open(file_path + '/label_' + str(i) + '.png')
        #file = file.convert('L')
        temp_u = (0.10+i*0.0025)*320
        sim_u.append(temp_u)
        file = np.array(file)
        file = range_scaling(file)
        sims.append(file)
    return sims, sim_u

def get_labels_bike(file_path):
    sims = []
    sim_u = []
    for i in range(34,63,1):
        file = Image.open(file_path + '/label_' + str(i) + '.png')
        sim_u.append(i)
        file = np.array(file)
        file = range_scaling(file)
        sims.append(file)
    return sims, sim_u


def range_scaling(array):
    minimum = np.min(array)
    maximum = np.max(array)
    new_array = (array - minimum)/(maximum-minimum)
    return new_array


def generate_batch(features, labels, u, batch_size=1, height=128, width=256, n_channels=3):
    index = random.choices(range(len(features)), k=batch_size)
    #index = [22]
    feature_batch = []
    target_batch = []
    for i in index:
        temp = features[i]+u[i]
        target_batch.append(labels[i])
        feature_batch.append(temp)
    feature_batch = np.array(feature_batch).astype(
        np.float32).reshape([batch_size, height, width, 1])
    target_batch = np.array(target_batch).astype(
        np.float32).reshape([batch_size, height, width, n_channels])
    return feature_batch, target_batch


def nin(x, num_units):
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = fc_layer(x, num_units)
    return tf.reshape(x, s[:-1]+[num_units])


def fc_layer(inputs, hiddens, nonlinearity=None, flat=False):
    input_shape = inputs.get_shape().as_list()
    if flat:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs_processed = tf.reshape(inputs, [-1, dim])
    else:
        dim = input_shape[1]
        inputs_processed = inputs

    weights = tf.Variable(tf.truncated_normal(
        shape=[dim, hiddens], dtype=tf.float32, stddev=0.1))
    biases = tf.Variable(tf.truncated_normal(
        shape=[hiddens], dtype=tf.float32, stddev=0.1))
    output_biased = tf.add(tf.matmul(inputs_processed, weights), biases)
    if nonlinearity is not None:
        output_biased = nonlinearity(ouput_biased)
    return output_biased


def concat_elu(x):
    axis = len(x.get_shape())-1
    return tf.nn.elu(tf.concat(values=[x, -x], axis=axis))


def set_nonlinearity(name):
    if name == 'concat_elu':
        return concat_elu
    elif name == 'elu':
        return tf.nn.elu
    elif name == 'concat_relu':
        return tf.nn.crelu
    elif name == 'relu':
        return tf.nn.relu
    else:
        raise('nonlinearity ' + name + ' is not supported')


def conv_layer(inputs, kernel_size, stride, num_features, nonlinearity=None):
    input_channels = int(inputs.get_shape()[3])
    weights = tf.Variable(tf.truncated_normal(shape=[kernel_size, kernel_size, input_channels, num_features],
                                              dtype=tf.float32, stddev=0.1))
    biases = tf.Variable(tf.truncated_normal(
        [num_features], dtype=tf.float32, stddev=0.1))
    conv = tf.nn.conv2d(inputs, weights, strides=[
                        1, stride, stride, 1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    if nonlinearity is not None:
        conv_biased = nonlinearity(conv_biased)
    return conv_biased


def transpose_conv_layer(inputs, kernel_size, stride, num_features, nonlinearity=None):
    input_channels = int(inputs.get_shape()[3])

    weights = tf.Variable(tf.truncated_normal(shape=[kernel_size, kernel_size, num_features, input_channels],
                                              dtype=tf.float32, stddev=0.1))
    biases = tf.Variable(tf.truncated_normal(
        shape=[num_features], dtype=tf.float32, stddev=0.1))
    batch_size = tf.shape(inputs)[0]
    output_shape = tf.stack([tf.shape(inputs)[0], tf.shape(
        inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_features])
    conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[
                                  1, stride, stride, 1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    if nonlinearity is not None:
        conv_biased = nonlinearity(conv_biased)
    shape = int_shape(inputs)
    conv_biased = tf.reshape(
        conv_biased, [shape[0], shape[1]*stride, shape[2]*stride, num_features])

    return conv_biased


def int_shape(x):
    shape = []
    for i in x.get_shape():
        shape.append(int(i))
    return shape


def res_block(x, a=None, filter_size=16, nonlinearity=concat_elu, keep_p=1.0, stride=1, gated=False):
    orig_x = x
    orig_x_int_shape = int_shape(x)
    if orig_x_int_shape[3] == 1:
        x_1 = conv_layer(x, 3, stride, filter_size)
    else:
        x_1 = conv_layer(nonlinearity(x), 3, stride, filter_size)
    if a is not None:
        shape_a = int_shape(a)
        shape_x_1 = int_shape(x_1)
        a = tf.pad(a, [[0, 0], [0, shape_x_1[1]-shape_a[1]],
                       [0, shape_x_1[2]-shape_a[2]], [0, 0]])
        x_1 += nin(nonlinearity(a), filter_size)
    x_1 = nonlinearity(x_1)
    if keep_p < 1.0:
        x_1 = tf.nn.dropout(x_1, keep_prob=keep_p)
    if not gated:
        x_2 = conv_layer(x_1, 3, 1, filter_size)
    else:
        x_2 = conv_layer(x_1, 3, 1, filter_size*2)
        x_2_1, x_2_2 = tf.split(axis=3, num_or_size_splits=2, value=x_2)
        x_2 = x_2_1 * tf.nn.sigmoid(x_2_2)

    if int(orig_x.get_shape()[2]) > int(x_2.get_shape()[2]):
        assert(int(orig_x.get_shape()[
               2]) == 2*int(x_2.get_shape()[2]), "res net block only supports stirde 2")
        orig_x = tf.nn.avg_pool(orig_x, [1, 2, 2, 1], [
                                1, 2, 2, 1], padding='SAME')

    # pad it
    out_filter = filter_size
    in_filter = int(orig_x.get_shape()[3])
    if out_filter != in_filter:
        orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [
                        (out_filter-in_filter), 0]])

    return orig_x + x_2
