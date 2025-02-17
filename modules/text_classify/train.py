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
print('size :', cfg.pad_size)

model_text_cls = Model(cfg)
model_text_cls.to(cfg.devices)
loss_func = nn.CrossEntropyLoss()

optimizer = optim.Adam(model_text_cls.parameters(), lr=cfg.learn_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

for epoch in range(cfg.num_epochs):
    for i, batch in enumerate(train_dataloader):
        label, data = batch
        data = torch.tensor(data).to(cfg.devices)
        label = torch.tensor(label, dtype=torch.int64).to(cfg.devices)

        optimizer.zero_grad()
        pred = model_text_cls.forward(data)
        loss_val = loss_func(pred, label)

        print("epoch is {}, ite is {}, val is {}".format(epoch, i, loss_val))
        loss_val.backward()
        optimizer.step()

    scheduler.step()
    if epoch % 10 == 0:
        torch.save(model_text_cls.state_dict(), "models/{}.pth".format(epoch))
