import tensorflow as tf


class Config(object):
    '''
    默认配置
    '''
    learning_rate = 0.01
    num_samples = 5000
    batch_size = 168
    encoder_len = 15
    decoder_len = 20
    embedding_dim = 50
    hidden_dim = 100
    train_dir = './baidu_zd_small.txt'
    dev_dir = './dev.txt'
    test_dir = './test.txt'
    model_name = './save_model/seq2seq.model'


config = Config()

tf.app.flags.DEFINE_float("learning_rate", config.learning_rate, "学习率")
tf.app.flags.DEFINE_integer("num_samples", config.num_samples, "采样损失函数的采样的样本数")
tf.app.flags.DEFINE_integer("batch_size", config.batch_size, "批处理的样本数量")
tf.app.flags.DEFINE_integer("encoder_len", config.encoder_len, "编码数据的长度")
tf.app.flags.DEFINE_integer("decoder_len", config.decoder_len, "解码数据的长度")
tf.app.flags.DEFINE_integer("embedding_dim", config.embedding_dim, "词嵌入惟独.")
tf.app.flags.DEFINE_string("train_dir", config.train_dir, "训练数据的路径")
tf.app.flags.DEFINE_string("dev_dir", config.dev_dir, "验证数据文件路径")
tf.app.flags.DEFINE_string("test_dir", config.test_dir, "测试数据文件路径")
tf.app.flags.DEFINE_boolean("train", False, "是否进行训练")  # true for prediction
tf.app.flags.DEFINE_boolean("predict", False, "是否进行预测")  # true for prediction
FLAGS = tf.app.flags.FLAGS


def main(_):
    print(FLAGS.learning_rate)

if __name__ == '__main__':
    tf.app.run()