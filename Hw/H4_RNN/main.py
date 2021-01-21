# main.py
import os

# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
from Hw.H4_RNN.data import TwitterDataset
from Hw.H4_RNN.model import LSTM_Net
from Hw.H4_RNN.preprocess import Preprocess
from Hw.H4_RNN.train import training
from Hw.H4_RNN.utils import load_training_data
import torch.utils.data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# define path
path_prefix = "./model"
data_dir = "/Users/jiangzhigang/Projects/DLAndML-Experiment/Data/H4_RNN"
training_label_path = os.path.join(data_dir, "training_label.txt")
training_nolabel_path = os.path.join(data_dir, "training_nolabel.txt")
testing_data_path = os.path.join(data_dir, "testing_data.txt")
w2v_all_model_path = os.path.join(path_prefix, 'w2v_all.model')
model_dir = path_prefix  # model directory for checkpoint model


# 定義句子長度、要不要固定 embedding、batch 大小、要訓練幾個 epoch、learning rate 的值、model 的資料夾路徑
sen_len = 20
fix_embedding = True  # fix embedding during training
batch_size = 128
epoch = 5
lr = 0.001
# model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model

print("loading data ...")  # 把 'training_label.txt' 跟 'training_nolabel.txt' 讀進來
train_x, y = load_training_data(training_label_path)

# 對 input 跟 labels 做預處理
preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_all_model_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
print(y)
y = preprocess.labels_to_tensor(y)

# 製作一個 model 的對象
model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device)  # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

# 把 data 分為 training data 跟 validation data（將一部份 training data 拿去當作 validation data）
X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]

# 把 data 做成 dataset 供 dataloader 取用
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)

# 把 data 轉成 batch of tensors
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=8)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=8)

# 開始訓練
training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)
