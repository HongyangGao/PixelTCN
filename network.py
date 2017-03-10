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
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        self.configure_networks()

    def configure_networks(self):
        self.train_reader, self.valid_reader = self.get_data_readers()
        with tf.variable_scope('pixel_cnn') as scope:
            self.train_preds, self.train_loss_op, self.train_miou, self.train_summary = self.build_network(
                'train', self.train_reader)
            scope.reuse_variables()
            self.valid_preds, self.valid_loss_op, self.valid_miou, self.valid_summary = self.build_network(
                'valid', self.valid_reader)
        optimizer = tf.train.AdamOptimizer(self.conf.learning_rate)
        self.train_op = optimizer.minimize(
            self.train_loss_op, name='global/train_op')
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)

    def get_data_readers(self):
        input_params = (
            self.sess, self.conf.data_dir, self.conf.train_list,
            (self.conf.height, self.conf.width), self.conf.class_num, 'train',
            self.data_format)
        train_reader = BatchDataReader(*input_params)
        input_params = (
            self.sess, self.conf.data_dir, self.conf.valid_list,
            (self.conf.height, self.conf.width), self.conf.class_num, 'valid',
            self.data_format)
        valid_reader = BatchDataReader(*input_params)
        return train_reader, valid_reader

    def build_network(self, name, data_reader):
        inputs, annotations = data_reader.next_batch(self.conf.batch)
        predictions = self.inference(inputs)
        losses = tf.losses.softmax_cross_entropy(
            annotations, predictions, scope=name+'/losses')
        loss_op = tf.reduce_mean(losses, name=name+'/loss_op')
        loss_summary = tf.summary.scalar(name+'/loss', loss_op)
        correct_prediction = tf.equal(
            tf.argmax(annotations, self.channel_axis),
            tf.argmax(predictions, self.channel_axis),
            name=name+'/correct_pred')
        accuracy_op = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), name=name+'/accuracy_op')
        accuracy_summary = tf.summary.scalar(name+'/accuracy', accuracy_op)
        m_iou, update_op = tf.contrib.metrics.streaming_mean_iou(
            predictions, annotations, self.conf.class_num, name=name+'/m_iou')
        m_iou_summary = tf.summary.scalar(name+'/m_iou', m_iou)
        summary = tf.summary.merge(
            [accuracy_summary, loss_summary, m_iou_summary])
        return predictions, loss_op, update_op, summary

    def inference(self, inputs):
        outputs = inputs
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
        return outputs

    def construct_down_block(self, inputs, name, down_outputs, first=False):
        num_outputs = self.conf.start_channel_num if first else 2 * \
            inputs.shape[self.channel_axis].value
        conv1 = conv2d(
            inputs, num_outputs, self.conv_size, name+'/conv1',
            self.conf.keep_prob, self.data_format)
        conv2 = conv2d(
            conv1, num_outputs, self.conv_size, name+'/conv2',
            self.conf.keep_prob, self.data_format)
        down_outputs.append(conv2)
        pool = pool2d(
            conv2, self.pool_size, name+'/pool', self.data_format)
        return pool

    def construct_bottom_block(self, inputs, name):
        num_outputs = inputs.shape[self.channel_axis].value
        conv1 = conv2d(
            inputs, 2*num_outputs, self.conv_size, name+'/conv1',
            self.conf.keep_prob, self.data_format)
        conv2 = conv2d(
            conv1, num_outputs, self.conv_size, name+'/conv2',
            self.conf.keep_prob, self.data_format)
        return conv2

    def construct_up_block(self, inputs, down_inputs, name, final=False):
        num_outputs = inputs.shape[self.channel_axis].value
        conv1 = dilated_conv(
            inputs, num_outputs, self.conv_size, name+'/conv1',
            self.axis, self.conf.keep_prob, self.data_format)
        conv1 = tf.concat(
            [conv1, down_inputs], self.channel_axis, name=name+'/concat')
        conv2 = conv2d(
            conv1, num_outputs, self.conv_size, name+'/conv2',
            self.conf.keep_prob, self.data_format)
        num_outputs = self.conf.class_num if final else num_outputs/2
        conv3 = conv2d(
            conv2, num_outputs, self.conv_size, name+'/conv3',
            self.conf.keep_prob, self.data_format)
        return conv3

    def train(self):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
        self.train_reader.start()
        self.valid_reader.start()
        for epoch_num in range(self.conf.max_epoch):
            if epoch_num % self.conf.test_step == 0:
                loss, _, summary = self.sess.run(
                    [self.valid_loss_op, self.valid_miou, self.valid_summary])
                self.save_summary(summary, epoch_num)
                print('----testing loss', loss)
            elif epoch_num % self.conf.summary_step == 0:
                loss, _, _, summary = self.sess.run(
                    [self.train_loss_op, self.train_op,
                    self.train_miou, self.train_summary])
                self.save_summary(summary, epoch_num)
            else:
                loss, _, _ = self.sess.run([
                    self.train_loss_op, self.train_op, self.train_miou])
                print('----training loss', loss)
            if epoch_num % self.conf.save_step == 0:
                self.save(epoch_num)
        self.train_reader.close()
        self.valid_reader.close()

    def save_summary(self, summary, step):
        print('---->summarying', step)
        self.writer.add_summary(summary, step)

    def test(self, step):
        print('---->testing', step)
        pass

    def predict(self, inputs):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
        dummy_annotations = np.empty(self.output_shape)
        #feed_dict = {self.inputs: inputs, self.annotations: dummy_annotations}
        predictions = self.sess.run(self.predictions)
        return predictions

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(conf.modeldir, conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path):
            print('------- no such checkpoint')
            return
        self.saver.restore(self.sess, model_path)
