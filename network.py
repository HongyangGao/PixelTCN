

class DilatedPixelCNN(object):

    def __init__(self, sess, conf, height, width, channel):
        self.sess = sess
        self.conf = conf
        self.height, self.width, self.channel = height, width, channel

        input_shap = [None, self, height, self.width, self.channel]

    def build_network(self):
        pass
    
    def train(self):
        pass

    def test(self):
        pass

    def generate(self):
        pass

    def save(self):
        pass

    def load(self):
        pass