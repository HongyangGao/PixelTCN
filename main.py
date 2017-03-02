import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from cifar10 import IMAGE_SIZE, inputs, maybe_download_and_extract


def configure():
    flags = tf.app.flags
    # training
    flags.DEFINE_float("max_epoch", 100000, "# of step in an epoch")
    flags.DEFINE_float("test_step", 100, "# of step to test a model")
    flags.DEFINE_float("save_step", 1000, "# of step to save a model")
    flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
    # data
    flags.DEFINE_string("dataset", "mnist", "Name of dataset [mnist, cifar]")
    flags.DEFINE_string("data_dir", "data", "Name of data directory")
    flags.DEFINE_string("sample_dir", "samples", "Sample directory")
    # Debug
    flags.DEFINE_boolean("is_train", True, "Training or testing")
    flags.DEFINE_string("log_level", "INFO", "Log level")
    flags.DEFINE_integer("random_seed", int(time.time()), "random seed")
    return flags.FLAGS


def prepare_data(dataset):
    DATA_DIR = os.path.join(conf.data_dir, conf.dataset)
    if conf.dataset == "mnist":
        mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
        next_train_batch = lambda x: mnist.train.next_batch(x)[0]
        next_test_batch = lambda x: mnist.test.next_batch(x)[0]
        height, width, channel = 28, 28, 1
        train_step_per_epoch = mnist.train.num_examples / conf.batch_size
        test_step_per_epoch = mnist.test.num_examples / conf.batch_size
    elif conf.dataset == "cifar":
        maybe_download_and_extract(DATA_DIR)
        images, labels = inputs(eval_data=False, 
            data_dir=os.path.join(DATA_DIR, 'cifar-10-batches-bin'), batch_size=conf.batch_size)
        height, width, channel = IMAGE_SIZE, IMAGE_SIZE, 3

# logging
logger = logging.getLogger()
logger.setLevel(conf.log_level)

# random seed
tf.set_random_seed(conf.random_seed)



