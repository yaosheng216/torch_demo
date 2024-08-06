import random
import torch
import torch.nn as nn
from torch import optim
from utils import showPlot, timeSince
from dataset import readLangs, SOS_token, EOS_token
from model import EncoderRNN, AttendDecoderRNN
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 11

teacher_forcing_ratio = 0.5

lang1 = "en"
lang2 = "cn"
path = "data/en-cn.txt"
input_lang, output_lang, pairs = readLangs(lang1, lang2, path)
print(input_lang.n_words)
print(input_lang.word2count)
print(output_lang.word2count)
print(output_lang.index2word)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for
            word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes,
                        dtype=torch.long,
                        device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def trainIters(encoder, decoder, n_iters,
               print_every=1000, plot_every=1000,
               save_every=100000, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]

    criterion = nn.NLLLoss()
    scheduler_encoder = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=1, gamma=0.95)
    scheduler_decoder = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=1, gamma=0.95)
    for item in range(1, n_iters + 1):
        training_pair = training_pairs[item - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor,
                     target_tensor,
                     encoder,
                     decoder,
                     encoder_optimizer,
                     decoder_optimizer,
                     criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if item % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, item / n_iters),
                                         item, item / n_iters * 100, print_loss_avg))

        if item % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if item % save_every == 0:
            torch.save(encoder.state_dict(), "models/encoder_{}.pth".format(item))
            torch.save(decoder.state_dict(), "models/decoder_{}.pth".format(item))

            scheduler_decoder.step()
            scheduler_encoder.step()

    showPlot(plot_losses)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() \
                                  < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the targer as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teaching forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttendDecoderRNN(hidden_size, output_lang.n_words,
                                max_len=MAX_LENGTH, dropout_p=0.1).to(device)
trainIters(encoder1, attn_decoder1, 1000000, save_every=20000, print_every=5000)
