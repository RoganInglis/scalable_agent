import sonnet as snt
import tensorflow as tf


def convnet(frame):
    with tf.variable_scope('convnet'):
        conv_out = frame
        for i, (num_ch, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
            # Downscale.
            conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
            conv_out = tf.nn.pool(
                conv_out,
                window_shape=[3, 3],
                pooling_type='MAX',
                padding='SAME',
                strides=[2, 2])

            # Residual block(s).
            for j in range(num_blocks):
                with tf.variable_scope('residual_%d_%d' % (i, j)):
                    block_input = conv_out
                    conv_out = tf.nn.relu(conv_out)
                    conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
                    conv_out = tf.nn.relu(conv_out)
                    conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
                    conv_out += block_input

    conv_out = tf.nn.relu(conv_out)
    conv_out = snt.BatchFlatten()(conv_out)

    conv_out = snt.Linear(256)(conv_out)
    conv_out = tf.nn.relu(conv_out)

    return conv_out