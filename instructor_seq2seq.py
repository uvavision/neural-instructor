from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import requests
# from Shape2D import get_shape2d_loader, render_canvas
from Shape2DInstructorData import get_shape2d_instructor_data_loader, render_canvas
from utils import adjust_lr, clip_gradient
from torch.distributions import Categorical
from PainterModel import Shape2DPainterNet, get_painter_model_prediction
from tqdm import tqdm
import numpy as np
import random


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


# https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/13516dfd905b0755c28c922e39722ce09ca4f689/misc/utils.py#L17
# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[ix]
            else:
                break
        out.append(txt)
    return out

# FIXME
def get_add_remove_canvas(prev_canvas, final_canvas):
    current_canvas_mask = (prev_canvas.data.sum(dim=2, keepdim=True) >= 0).repeat(1, 1, 4)
    add_obj_canvas = final_canvas.data.clone()
    add_obj_canvas[current_canvas_mask] = -1
    remove_obj_canvas = prev_canvas.data.clone()
    # final_canvas_mask = (final_canvas.data.sum(dim=2, keepdim=True) >= 0).repeat(1, 1, 4)
    # remove_obj_mask = current_canvas_mask & final_canvas_mask
    remove_obj_mask = ((prev_canvas.data == final_canvas.data).sum(dim=2, keepdim=True) == 4).repeat(1, 1, 4)
    remove_obj_canvas[remove_obj_mask] = -1
    return add_obj_canvas, remove_obj_canvas


# https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/misc/utils.py#L39
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target):
        # truncate to the same size
        batch_size = input.size(0)
        target = target[:, :input.size(1)]
        # TODO
        num_non_zeros = (target.data > 0).sum(dim=1) # B
        mask = input.data.new(target.size()).fill_(0)
        for i in range(mask.size(0)):
            mask[i, :num_non_zeros[i]+1] = 1
        mask = Variable(mask, requires_grad=False).view(-1, 1)
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


# convert canvas tensor (with batch) to sequence of objs
def canvas2seq(canvas):
    canvas_data = canvas.data
    out = canvas_data.new(canvas.size(0), 25, 4).fill_(-1)
    obj_mask = canvas_data.sum(dim=2) >= 0
    for i in range(canvas.size(0)):
        if obj_mask[i].sum() > 0:
            indices = torch.nonzero(obj_mask[i])
            objs = torch.gather(canvas_data[i], 0, indices.repeat(1, 4))
            out[i, :objs.size(0)] = objs
    return Variable(out)


class CanvasEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_size = 64
        self.input_size = 64
        # https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/models/AttModel.py#L43
        # self.embed = nn.Embedding(4, self.input_size)
        self.obj_embed = nn.Linear(4, self.input_size)
        # https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/models/AttModel.py#L372
        self.lstm_cell = nn.LSTMCell(self.input_size, self.rnn_size)

    def forward(self, canvas):
        # TODO early stop when column contains all zero
        # http://pytorch.org/docs/master/nn.html#torch.nn.LSTMCell
        obj_seqs = canvas2seq(canvas)
        batch_size = obj_seqs.size(0)
        h = Variable(torch.zeros(batch_size, self.rnn_size).cuda())
        c = Variable(torch.zeros(batch_size, self.rnn_size).cuda())
        for i in range(obj_seqs.size(1)):
            if (obj_seqs[:, i].data.sum(dim=1) >= 0).sum() == 0:
                break
            input = self.obj_embed(obj_seqs[:, i])
            h, c = self.lstm_cell(input, (h, c))
        # only want the last hidden state
        return h


class InstLM(nn.Module):
    def __init__(self, vocab_size, max_seq_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.rnn_size = 128
        self.input_size = 64
        # https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/models/AttModel.py#L43
        self.embed = nn.Embedding(vocab_size + 1, self.input_size)
        # https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/models/AttModel.py#L372
        self.lstm_cell = nn.LSTMCell(self.input_size, self.rnn_size)
        self.fc_out = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.canvas_encoder = CanvasEncoder()
        self.use_canvas_concat = False
        if self.use_canvas_concat:
            self.canvas_trans = nn.Linear(self.canvas_encoder.rnn_size * 2, self.input_size)
        self.target_obj_embed = nn.Linear(4, self.input_size)

    def forward(self, inst, prev_canvas, final_canvas, extra=None):
        # http://pytorch.org/docs/master/nn.html#torch.nn.LSTMCell
        target_obj_embedding = self.target_obj_embed(extra[0].float())
        prev_canvas_encoding = self.canvas_encoder(prev_canvas)
        final_canvas_encoding = self.canvas_encoder(final_canvas)
        if self.use_canvas_concat:
            canvas_encoding = torch.cat([prev_canvas_encoding, final_canvas_encoding], dim=1)
            canvas_encoding = self.canvas_trans(canvas_encoding)
        else:
            canvas_encoding = prev_canvas_encoding + final_canvas_encoding
        batch_size = inst.size(0)
        h = Variable(torch.zeros(batch_size, self.rnn_size).cuda())
        c = Variable(torch.zeros(batch_size, self.rnn_size).cuda())
        h, c = self.lstm_cell(canvas_encoding, (h, c))
        h, c = self.lstm_cell(target_obj_embedding, (h, c))
        outputs = []
        # inst: [0, a, b, c, 0]
        for i in range(inst.size(1) - 1):
            if i >= 1 and inst[:, i].data.sum() == 0:
                break
            xt = self.embed(inst[:, i])
            h, c = self.lstm_cell(xt, (h, c))
            output = F.log_softmax(self.fc_out(h), dim=1) # Bx(vocab_size+1)
            outputs.append(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1) # Bx(seq_len)x(vocab_size+1)

    # https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/models/AttModel.py#L151
    def sample(self, prev_canvas, final_canvas, extra=None):
        assert not self.embed.training
        assert not self.lstm_cell.training
        assert not self.fc_out.training
        batch_size = prev_canvas.size(0)
        sample_max = True
        temperature = 0.8
        seq = []
        target_obj_embedding = self.target_obj_embed(extra[0].float())
        prev_canvas_encoding = self.canvas_encoder(prev_canvas)
        final_canvas_encoding = self.canvas_encoder(final_canvas)
        if self.use_canvas_concat:
            canvas_encoding = torch.cat([prev_canvas_encoding, final_canvas_encoding], dim=1)
            canvas_encoding = self.canvas_trans(canvas_encoding)
        else:
            canvas_encoding = prev_canvas_encoding + final_canvas_encoding
        h = Variable(torch.zeros(batch_size, self.rnn_size).cuda())
        c = Variable(torch.zeros(batch_size, self.rnn_size).cuda())
        h, c = self.lstm_cell(canvas_encoding, (h, c))
        h, c = self.lstm_cell(target_obj_embedding, (h, c))
        for t in range(self.max_seq_length + 1):
            if t == 0:
                it = torch.zeros(batch_size).long().cuda()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu()  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                    it = torch.multinomial(prob_prev, 1).cuda()
                    sampleLogprobs = logprobs.gather(1, Variable(it, volatile=True))  # gather the logprobs at sampled positions
                    it = it.view(-1).long()
            if t >= 1:
                seq.append(it)
            xt = self.embed(Variable(it, volatile=True))
            h, c = self.lstm_cell(xt, (h, c))
            logprobs = F.log_softmax(self.fc_out(h), dim=1) # Bx(vocab_size+1)
        return torch.t(torch.stack(seq)).cpu()


val_loader = get_shape2d_instructor_data_loader(split='val', batch_size=20, shuffle=False)
train_loader = get_shape2d_instructor_data_loader(split='val', batch_size=args.batch_size)
assert train_loader.dataset.vocab_size == val_loader.dataset.vocab_size
assert train_loader.dataset.max_seq_length == val_loader.dataset.max_seq_length

val_loader_iter = iter(val_loader)

model = InstLM(train_loader.dataset.vocab_size, train_loader.dataset.max_seq_length)
# model.load_state_dict(torch.load('instructor_lm/model_9.pth'))
if model.use_canvas_concat:
    model.load_state_dict(torch.load('instructor_seq2seq_target_canvas_concat/model_20.pth'))
else:
    model.load_state_dict(torch.load('instructor_seq2seq_target/model_20.pth'))
model.cuda()
loss_fn = LanguageModelCriterion()

painter_model = Shape2DPainterNet(train_loader.dataset.vocab_size)
painter_model.load_state_dict(torch.load('painter_model_new_add_remove///model_20.pth'))
painter_model.cuda()
painter_model.eval()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)


def eval_sample():
    model.eval()
    val_data = next(val_loader_iter)
    val_raw_data = val_data[-1]
    data = [Variable(_, volatile=True).cuda() for _ in val_data[:-1]]
    prev_canvas, final_canvas, inst, target_obj, act = data
    ref_obj = None
    add_canvas, remove_canvas = get_add_remove_canvas(prev_canvas, final_canvas)
    add_canvas_objs = [train_loader.dataset.decode_canvas(add_canvas[i]) for i in range(add_canvas.size(0))]
    remove_canvas_objs = [train_loader.dataset.decode_canvas(remove_canvas[i]) for i in range(remove_canvas.size(0))]
    samples = model.sample(prev_canvas, final_canvas, (target_obj, ref_obj, False))
    act_prediction, target_obj_prediction = get_painter_model_prediction(painter_model, samples, prev_canvas)
    samples = decode_sequence(train_loader.dataset.ix_to_word, samples)
    print(((target_obj_prediction[:, 2:] == target_obj.data[:, 2:]).sum(dim=1) == 2).sum())
    for i in range(prev_canvas.size(0)):
        val_raw_data[i]['predicted_instruction'] = '{} - {} ({} {} {} {} -> {} {} {} {}) [seq2seq]'.format(
            i, samples[i],
            train_loader.dataset.num2color[target_obj.data[i, 0]],
            train_loader.dataset.num2shape[target_obj.data[i, 1]],
            target_obj.data[i, 2], target_obj.data[i, 3],
            train_loader.dataset.num2color[target_obj_prediction[i, 0]],
            train_loader.dataset.num2shape[target_obj_prediction[i, 1]],
            target_obj_prediction[i, 2], target_obj_prediction[i, 3]
        )
    for i, d in enumerate(val_raw_data):
        d['prev_canvas'] = render_canvas(d['prev_canvas']).replace(' ', '')
        d['final_canvas'] = render_canvas(d['final_canvas']).replace(' ', '')
        d['add_canvas_data'] = render_canvas(add_canvas_objs[i]).replace(' ', '')
        d['remove_canvas_data'] = render_canvas(remove_canvas_objs[i]).replace(' ', '')
    r = requests.post("http://deep.cs.virginia.edu:5001/new_instruction", json=val_raw_data)
    print('request sent')


def train(epoch):
    model.train()
    adjust_lr(optimizer, epoch, args.lr, decay_rate=0.2)
    for batch_idx, data in enumerate(train_loader):
        raw_data = data[-1]
        data = [Variable(_, requires_grad=False).cuda() for _ in data[:-1]]
        prev_canvas, final_canvas, inst, target_obj, act = data
        ref_obj = None
        optimizer.zero_grad()
        loss = loss_fn(model(inst, prev_canvas, final_canvas, (target_obj, ref_obj, True)), inst[:, 1:])
        loss.backward()
        clip_gradient(optimizer, 0.1)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f})'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))
    torch.save(model.state_dict(), 'instructor_seq2seq_target_canvas_concat/model_{}.pth'.format(epoch))


if __name__ == '__main__':
    eval_sample()
    for epoch in range(1, args.epochs + 1):
        train(epoch)



