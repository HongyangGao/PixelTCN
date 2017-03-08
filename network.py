import os
import tensorflow as tf
from data_reader import BatchDataReader
from ops import conv2d, pool2d, dilated_conv


class DilatedPixelCNN(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.conv_size = (3, 3)
        self.pool_size = (2, 2)
        if conf.use_gpu:
            self.data_format = 'NCHW'
            self.axis, self.channel_axis = (2, 3), 1
            self.input_shape = [
                conf.batch, conf.channel, conf.height, conf.width]
            self.output_shape = [
                conf.batch, conf.class_num, conf.height, conf.width]
        else:
            self.data_format = 'NHWC'
            self.axis, self.channel_axis = (1, 2), 3
            self.input_shape = [
                conf.batch, conf.height, conf.width, conf.channel]
            self.output_shape = [
                conf.batch, conf.height, conf.width, conf.class_num]
        input_params = (
            self.sess, conf.data_dir, conf.train_list,
            (conf.height, conf.width), self.data_format)
        self.data_reader = BatchDataReader(*input_params)
        self.inputs, self.annotations = self.data_reader.next_batch(
            self.conf.batch)
        self.build_network()
        tf.set_random_seed(conf.random_seed)
        sess.run(tf.global_variables_initializer())
        # save point configure
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)

    def build_network(self):
        outputs = self.inputs
        down_outputs = []
        for layer_index in range(self.conf.network_depth-1):
            is_first = True if not layer_index else False
            name = 'down%s' % layer_index
            outputs = self.construct_down_block(
                outputs, name, down_outputs, first=is_first)
        outputs = self.construct_bottom_block(outputs, 'bottom')
        for layer_index in range(self.conf.network_depth-2, -1, -1):
            is_final = True if layer_index == 0 else False
            name = 'up%s' % layer_index
            down_inputs = down_outputs[layer_index]
            outputs = self.construct_up_block(
                outputs, down_inputs, name, final=is_final)
        self.prediction = outputs
        losses = tf.losses.softmax_cross_entropy(
            self.annotations, self.prediction, scope='losses')
        self.loss_op = tf.reduce_mean(losses, name='loss_op')
        tf.summary.scalar('loss', self.loss_op)
        correct_prediction = tf.equal(
            tf.argmax(self.annotations, self.channel_axis),
            tf.argmax(self.prediction, self.channel_axis),
            name='accuracy/correct_pred')
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
            name='accuracy/accuracy')
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(
            self.conf.logdir + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.conf.logdir + '/test')
        self.train_op = tf.train.AdamOptimizer(
            self.conf.learning_rate).minimize(self.loss_op, name='train_op')

    def construct_down_block(self, inputs, name, down_outputs, first=False):
        num_outputs = self.conf.start_channel_num if first else 2 * \
            inputs.shape[self.channel_axis].value
        conv1 = conv2d(
            inputs, num_outputs, self.conv_size, name+'/conv1',
            self.data_format)
        conv2 = conv2d(
            conv1, num_outputs, self.conv_size, name+'/conv2',
            self.data_format)
        down_outputs.append(conv2)
        pool = pool2d(
            conv2, self.pool_size, name+'/pool', self.data_format)
        return pool

    def construct_bottom_block(self, inputs, name):
        num_outputs = inputs.shape[self.channel_axis].value
        conv1 = conv2d(
            inputs, 2*num_outputs, self.conv_size, name+'/conv1',
            self.data_format)
        conv2 = conv2d(
            conv1, num_outputs, self.conv_size, name+'/conv2',
            self.data_format)
        return conv2

    def construct_up_block(self, inputs, down_inputs, name, final=False):
        num_outputs = inputs.shape[self.channel_axis].value
        conv1 = dilated_conv(
            inputs, num_outputs, self.conv_size, name+'/conv1',
            self.axis, self.data_format)
        conv1 = tf.concat(
            [conv1, down_inputs], self.channel_axis, name=name+'/concat')
        conv2 = conv2d(
            conv1, num_outputs, self.conv_size, name+'/conv2',
            self.data_format)
        num_outputs = self.conf.class_num if final else num_outputs/2
        conv3 = conv2d(
            conv2, num_outputs, self.conv_size, name+'/conv3',
            self.data_format)
        return conv3

    def train(self):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
        self.data_reader.start()
        for epoch_num in range(self.conf.max_epoch):
            loss, _ = self.sess.run([self.loss_op, self.train_op])
            if epoch_num % self.conf.save_step == 0:
                self.save(epoch_num)
            if epoch_num % self.conf.test_step == 0:
                self.test()
            if epoch_num % self.conf.summary_step == 0:
                self.save_summary(epoch_num)
        self.data_reader.close()

    def save_summary(self, step):
        print('---->summarying', step)
        summary = self.sess.run(self.merged_summary)
        self.train_writer.add_summary(summary, step)

    def test(self, step):
        print('---->testing', step)
        pass

    def predict(self):
        pass

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(conf.modeldir, conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path):
            print('------- no such checkpoint')
            return
        self.saver.restore(self.sess, model_path)
