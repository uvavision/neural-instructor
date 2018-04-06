from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from Shape2D import get_shape2d_loader
from utils import adjust_lr, clip_gradient


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


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, output, target):
        # truncate to the same size
        target = target[:, :output.size(1)]
        num_non_zeros = (target.data > 0).sum(dim=1) # B
        mask = output.data.new(target.size()).fill_(0)
        for i in range(mask.size(0)):
            mask[i, :num_non_zeros[i]+1] = 1
        mask = Variable(mask, requires_grad=False).view(-1, 1)
        output = to_contiguous(output).view(-1, output.size(2))
        target = to_contiguous(target).view(-1, 1)
        output = - output.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class InstLM(nn.Module):
    def __init__(self, vocab_size, max_seq_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.rnn_size = 64
        self.input_size = 64
        # https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/models/AttModel.py#L43
        self.embed = nn.Embedding(vocab_size + 1, self.input_size)
        # https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/models/AttModel.py#L372
        self.lstm_cell = nn.LSTMCell(self.input_size, self.rnn_size)
        self.fc_out = nn.Linear(self.rnn_size, self.vocab_size + 1)

    def forward(self, inst):
        # http://pytorch.org/docs/master/nn.html#torch.nn.LSTMCell
        batch_size = inst.size(0)
        h = Variable(torch.zeros(batch_size, self.rnn_size).cuda())
        c = Variable(torch.zeros(batch_size, self.rnn_size).cuda())
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
    def sample(self, batch_size):
        assert not self.embed.training
        assert not self.lstm_cell.training
        assert not self.fc_out.training
        sample_max = False
        temperature = 0.8
        seq = []
        h = Variable(torch.zeros(batch_size, self.rnn_size).cuda())
        c = Variable(torch.zeros(batch_size, self.rnn_size).cuda())
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

train_loader = get_shape2d_loader(split='train', batch_size=args.batch_size)
test_loader = get_shape2d_loader(split='val', batch_size=args.batch_size)
assert train_loader.dataset.vocab_size == test_loader.dataset.vocab_size
assert train_loader.dataset.max_seq_length == test_loader.dataset.max_seq_length

# sample_loader = get_shape2d_loader(split='sample', batch_size=2)
# print(len(sample_loader.dataset))

model = InstLM(train_loader.dataset.vocab_size, train_loader.dataset.max_seq_length)
model.cuda()
loss_fn = LanguageModelCriterion()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)


def train(epoch):
    adjust_lr(optimizer, epoch, args.lr, decay_rate=0.2)
    for batch_idx, data in enumerate(train_loader):
        data = [Variable(_, requires_grad=False).cuda() for _ in data]
        prev_canvas, inst, next_obj, final_canvas, ref_obj = data
        optimizer.zero_grad()
        loss = loss_fn(model(inst), inst[:, 1:])
        loss.backward()
        clip_gradient(optimizer, 0.1)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f})'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))
            model.eval()
            samples = model.sample(batch_size=10)
            samples = decode_sequence(train_loader.dataset.ix_to_word, samples)
            for i in range(10):
                print(samples[i])
            model.train()


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)

