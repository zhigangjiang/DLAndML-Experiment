from Hw.H8_Seq2Seq.config import configurations
from Hw.H8_Seq2Seq.train import train_process
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 判斷是用 CPU 還是 GPU 執行運算
config = configurations()

if not os.path.isdir(config.store_model_path):
    os.mkdir(config.store_model_path)


print('config:\n', vars(config))
train_losses, val_losses, bleu_scores = train_process(config, device)
