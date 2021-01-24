# main.py
import os

# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
from Hw.H4_RNN.data import TwitterDataset
from Hw.H4_RNN.model import LSTM_Net
from Hw.H4_RNN.preprocess import Preprocess
from Hw.H4_RNN.train import training
from Hw.H4_RNN.utils import *
import torch.utils.data
import argparse
import ast
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn

parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
parser.add_argument("--mode", choices=['train', 'continue', 'evaluate', 'test'], required=True, type=str,
                    help="the run mode", dest="mode")
parser.add_argument("--all_train", default=False, type=ast.literal_eval, help="is train dataset: train and evaluate.",
                    dest="all_train")
parser.add_argument("--visible_device", default=0, type=int, help="visible device",
                    dest="visible_device")

parser.add_argument("--data_dir", required=True, type=str, help="the dataset root dir", dest="data_dir")
parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str, help="the output checkpoints dir",
                    dest="checkpoint_dir")
parser.add_argument("--lr", default=0.001, type=float, help="the learning rate of adama.", dest="lr")
parser.add_argument("--batch_size", default=128, type=int, help="batch_size", dest="batch_size")
parser.add_argument("--checkpoint_path", default="", type=str, help="the output checkpoints path",
                    dest="checkpoint_path")
args = parser.parse_args()

print("arguments:")
for arg in vars(args):
    print(arg, ":", getattr(args, arg))

print("-" * 100)

mode = args.mode
all_train = args.all_train
data_dir = args.data_dir
visible_device = args.visible_device

lr = float(args.lr)
batch_size = int(args.batch_size)
checkpoint_dir = args.checkpoint_dir
checkpoint_path = args.checkpoint_path
start_epoch = 0
epoch = 500
best_acc = 0

torch.cuda.set_device(visible_device)

if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)

if mode == "train" and len([f for f in os.listdir(checkpoint_dir) if f.__contains__(".pth")]) > 0:
    ans = input("checkpoints--0 is not empty, do you cover? [y/n]")
    if ans != 'y':
        exit(0)

# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))

########################################################################################################################

# define path
training_label_path = os.path.join(data_dir, "training_label.txt")
training_nolabel_path = os.path.join(data_dir, "training_nolabel.txt")
testing_data_path = os.path.join(data_dir, "testing_data.txt")
w2v_all_model_path = os.path.join(checkpoint_dir, 'w2v_all.model')

# 定義句子長度、要不要固定 embedding、batch 大小、要訓練幾個 epoch、learning rate 的值、model 的資料夾路徑
sen_len = 20
fix_embedding = True  # fix embedding during training

print("loading data ...")  # 把 'training_label.txt' 跟 'training_nolabel.txt' 讀進來

# 對 input 跟 labels 做預處理
train_x, train_y = load_training_data(training_label_path)
test_x = load_testing_data(testing_data_path)

preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_all_model_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
train_y = preprocess.labels_to_tensor(train_y)
train_x_, val_x_, train_y_, val_y_ = train_x[:180000], train_x[180000:], train_y[:180000], train_y[180000:]


# 製作一個 model 的對象
model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device)  # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）
optimizer = optim.Adam(model.parameters(), lr=lr)
loss = nn.BCELoss()

if mode != "train":
    checkpoint_path = checkpoint_path if len(checkpoint_path) != 0 else get_best_checkpoint_path(checkpoint_dir)
    print("load checkpoint_path:{}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)  # 加载断点
    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch']  # 设置开始的epoch
    best_acc = checkpoint['best_acc']

if mode == "test":

    preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_all_model_path)
    embedding = preprocess.make_embedding(load=True)
    test_x = preprocess.sentence_word2idx()
    test_dataset = TwitterDataset(x=test_x, y=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    prediction = test(model, test_loader, device)
    with open(os.path.join(checkpoint_dir, "predict.csv"), 'w') as f:
        f.write('id,label\n')
        for i, y in enumerate(prediction):
            print(y)
            f.write('{},{}\n'.format(i, y))

else:
    val_dataset = TwitterDataset(x=val_x_, y=val_y_)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if mode == "train" or mode == "continue":
        if all_train:
            train_dataset = TwitterDataset(x=train_x_, y=train_y_)
        else:
            train_dataset = TwitterDataset(x=train_x, y=train_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        training(start_epoch, epoch, optimizer, checkpoint_dir, train_loader, val_loader, model, loss, best_acc,
                 all_train,
                 device)
    else:
        val_loss, val_acc = evaluate(start_epoch, model, val_loader, loss, device, None, is_train=False)
