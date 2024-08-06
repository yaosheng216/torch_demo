import random
import torch
import torch.nn as nn
from torch import optim
from dataset import readLangs, SOS_token, EOS_token, MAX_LENGTH
from model import EncoderRNN, AttendDecoderRNN
from utils import timeSince
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = MAX_LENGTH + 1

lang1 = "en"
lang2 = "cn"
path = "data/en-cn.txt"
input_lang, output_lang, pairs = readLangs(lang1, lang2, path)
print(len(pairs))
print(input_lang.n_words)
print(input_lang.index2word)

print(output_lang.n_words)
print(output_lang.index2word)


def listToTensor(input_lang, data):
    indexes_in = [input_lang.word2index[word] for word in data.split(" ")]
    indexes_in.append(EOS_token)
    input_tensor = torch.tensor(indexes_in,
                                dtype=torch.long,
                                device=device).view(-1, 1)
    return input_tensor


def tensorsFromPair(pair):
    input_tensor = listToTensor(input_lang, pair[0])
    output_tensor = listToTensor(output_lang, pair[1])
    return (input_tensor, output_tensor)


hidden_size = 256
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttendDecoderRNN(hidden_size, output_lang.n_words, max_len=MAX_LENGTH, dropout_p=0.1).to(device)

encoder.load_state_dict(torch.load("models/encoder_1000000.pth"))
decoder.load_state_dict(torch.load("models/decoder_1000000.pth"))
n_iters = 10

train_sen_pairs = [
    random.choice(pairs) for i in range(n_iters)
]
training_pairs = [
    tensorsFromPair(train_sen_pairs[i]) for i in range(n_iters)
]

for i in range(n_iters):
    input_tensor, output_tensor = training_pairs[i]
    encoder_hidden = encoder.initHidden()
    input_len = input_tensor.size(0)
    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    for ei in range(input_len):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_hidden = encoder_hidden
    decoder_input = torch.tensor([[SOS_token]], device=device)
    use_teacher_forcing = True if random.random() < 0.5 else False
    decoder_words = []
    for di in range(MAX_LENGTH):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        topV, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        if topi.item() == EOS_token:
            decoder_words.append("<EOS>")
            break
        else:
            decoder_words.append(output_lang.index2word[topi.item()])

    print(train_sen_pairs[i][0])
    print(train_sen_pairs[i][1])
    print(decoder_words)
