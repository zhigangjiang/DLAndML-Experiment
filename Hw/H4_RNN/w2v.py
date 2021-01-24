# w2v.py
# 這個 block 是用來訓練 word to vector 的 word embedding
# 注意！這個 block 在訓練 word to vector 時是用 cpu，可能要花到 10 分鐘以上
# 这个不在RNN训练中

import os
from gensim.models import word2vec
from Hw.H4_RNN.utils import *
import argparse


parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str, help="the output checkpoints dir",
                    dest="checkpoint_dir")
parser.add_argument("--data_dir", required=True, type=str, help="the dataset root dir", dest="data_dir")
args = parser.parse_args()
print("arguments:")
for arg in vars(args):
    print(arg, ":", getattr(args, arg))

print("-" * 100)

checkpoint_dir = args.checkpoint_dir
data_dir = args.data_dir

if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)


def train_word2vec(x):
    print("train_word2vec length:{}".format(len(x)))
    # 訓練 word to vector 的 word embedding
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model


training_label_path = os.path.join(data_dir, "training_label.txt")
training_nolabel_path = os.path.join(data_dir, "training_nolabel.txt")
testing_data_path = os.path.join(data_dir, "testing_data.txt")
w2v_all_model_path = os.path.join(checkpoint_dir, 'w2v_all.model')

if __name__ == "__main__":
    print("loading training data ...")
    train_x, y = load_training_data(training_label_path)
    print("x length:{} y length:{}".format(len(train_x), len(y)))

    print("loading training no label data ...")
    train_x_no_label = load_training_data(training_nolabel_path)
    print("x length:{} y".format(len(train_x_no_label)))

    print("loading testing data ...")
    test_x = load_testing_data(testing_data_path)
    print("x length:{} y".format(len(test_x)))

    model = train_word2vec(train_x + train_x_no_label + test_x)
    # model = train_word2vec(train_x + test_x)

    print("saving model ...")
    # model.save(os.path.join(path_prefix, 'model/w2v_all.model'))
    model.save(w2v_all_model_path)
