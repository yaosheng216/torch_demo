import torch
import torch.nn as nn
from torch import optim
from model import Model
from dataset import data_loader, text_ClS
from config import Config


cfg = Config()
data_path = "sources/weibo_senti_100k.csv"
data_stop_path = "sources/hit_stopword"
dict_path = "sources/dict"

dataset = text_ClS(dict_path, data_path, data_stop_path)
train_dataloader = data_loader(dataset, cfg)

cfg.pad_size = dataset.max_seq_len
print(cfg.pad_size)

model_text_cls = Model(cfg)
model_text_cls.to(cfg.devices)
model_text_cls.load_state_dict(torch.load("models/10.pth"))


for i, batch in enumerate(train_dataloader):
    label, data = batch
    data = torch.tensor(data).to(cfg.devices)
    label = torch.tensor(label, dtype=torch.int64).to(cfg.devices)
    pred_softmax = model_text_cls.forward(data)
    pred = torch.argmax(pred_softmax, dim=1)
    out = torch.eq(pred, label)
    print(out.sum() * 1.0 / pred.size()[0])
