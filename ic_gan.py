#semi-supervised GAN for IC design by yang xulei, June.14, 2018
#forked from bruno's semi-supervised GAN for image classification 
import tensorflow as tf
import nn  # OpenAI implemetation of weightnormalization (Salimans & Kingma)

init_kernel = tf.random_normal_initializer(mean=0, stddev=0.05)


def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)


def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def discriminator(inp, is_training, init=False, reuse=False, getter =None):
    with tf.variable_scope('discriminator_model', reuse=reuse,custom_getter=getter):


        # D(x)
        name_x = 'x_layer_1'
        with tf.variable_scope(name_x):
            x = tf.layers.dense(inp,
                          units=32,
                          kernel_initializer=init_kernel,
                          name='fc')
            x = leakyReLu(x)
            x = tf.layers.dropout(x, rate=0.2, name='dropout')


        name_y = 'y_fc_1'
        with tf.variable_scope(name_y):
            y = tf.layers.dense(x,
                                8,
                                kernel_initializer=init_kernel)
            y = leakyReLu(y)
            y = tf.layers.dropout(y, rate=0.2, name='dropout')

        intermediate_layer = y

        name_y = 'y_fc_logits'
        with tf.variable_scope(name_y):
            logits = tf.layers.dense(y,
                                     2,
                                     kernel_initializer=init_kernel)

        return logits, intermediate_layer


def generator(z_seed, is_training, init=False,reuse=False):
    with tf.variable_scope('generator_model', reuse=reuse):
        inp = z_seed
        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(inp,
                                  units=32,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net)
        """
        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=128,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net)
        """
        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=5,
                                  kernel_initializer=init_kernel,
                                  name='fc')


        return net
