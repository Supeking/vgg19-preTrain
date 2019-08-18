import tensorflow as tf
import numpy as np
import utils
import os
import genData
import cv2

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'


class Vgg19():
    def __init__(self):
        self.lr = 1e-5
        self.batchsize = 10
        self.checkpoint_dir = 'model'
        self.model_name = 'insulator'
        self.sess = tf.Session()
        self.build()

    def vgg_net(self, weights, image):
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )
        net = {}
        current = image
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current, name=name)
            elif kind == 'pool':
                current = utils.avg_pool_2x2(current)
            net[name] = current
        return net

    def inference(self, image):
        """
        Semantic segmentation network definition
        :param image: input image. Should have values in range 0-255
        :param keep_prob:
        :return:
        """
        print("setting up vgg initialized conv layers ...")
        model_data = utils.get_model_data('./model_zoo', MODEL_URL)

        mean = model_data['normalization'][0][0][0]
        # mean_pixel = np.mean(mean, axis=(0, 1))

        weights = np.squeeze(model_data['layers'])
        # processed_image = utils.process_image(image, mean_pixel)

        with tf.variable_scope("inference"):
            image_net = self.vgg_net(weights, image)
            net = image_net["conv5_2"]
            map = tf.layers.conv2d(net, 2, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
            out = tf.reduce_mean(map, [1, 2])
        return map, out

    def build(self):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, 448, 448, 3])
        self.labels = tf.placeholder(dtype=tf.int64, shape=[None])
        self.maps, self.outs = self.inference(self.inputs)
        self.accr = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.outs, 1), self.labels), tf.float16))
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.outs, labels=self.labels))
        self.optm = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        # 模型保存
        self.saver = tf.train.Saver(max_to_keep=2)
        # 初始化
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print('build model success!')

    def train(self, Epoch=2000, display=200):
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        read_P = genData.readImg_P()
        read_N = genData.readImg_N()
        loss_avg = []
        accr_avg = []
        for eph in range(Epoch):
            label = np.random.randint(0, 2, self.batchsize)
            im = []
            for lab in label:
                if lab==0:
                    im.append(next(read_N))
                else:
                    im.append(next(read_P))
            img = np.array(im)

            feed = {self.inputs: img, self.labels: label}
            _, loss, accr = self.sess.run([self.optm, self.loss, self.accr], feed_dict=feed)
            loss_avg.append(loss)
            # print(loss)
            accr_avg.append(accr)
            global_step = (eph+1)*self.batchsize
            if global_step % display==0:
                loss_avg = np.mean(np.array(loss_avg))
                accr_avg = np.mean(np.array(accr_avg))
                print('Epoch:{} loss:{:.5f} accr:{:.1%}'.format(global_step, loss_avg, accr_avg))
                loss_avg = []
                accr_avg = []
                self.save(self.checkpoint_dir, global_step)

    def test(self):
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        read_P = genData.readImg_P()
        for _ in range(50):
            im = next(read_P)
            gray_im = cv2.cvtColor(np.uint8(im * 255), cv2.COLOR_BGR2GRAY)
            img = im[None, ...]
            out, map, accr = self.sess.run([self.outs, self.maps, self.accr], feed_dict={self.inputs: img, self.labels:[1]})
            Map = map[0, ..., 1]
            im_map = np.uint8(((Map-np.min(Map))/(np.max(Map)-np.min(Map)))*255)
            im_map = cv2.resize(im_map, (448, 448))
            imm = np.hstack((gray_im, im_map))
            cv2.imshow('x', imm)
            cv2.waitKey(0)
            print('-')

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(
            checkpoint_dir, self.model_name), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            return True
        else:
            return False


if __name__ == '__main__':
    model = Vgg19()
    train = 0
    if train:
        model.train()
    else:
        model.test()
