from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import requests
from Shape2D import get_shape2d_loader, render_canvas
from utils import adjust_lr, clip_gradient
from torch.distributions import Categorical
from tqdm import tqdm


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


def compute_diff_canvas(canvas1, canvas2):
    mask = (canvas1.sum(dim=2, keepdim=True) >= 0).repeat(1, 1, 4)
    diff_canvas = Variable(canvas2.data.clone())
    diff_canvas[mask] = -1
    return diff_canvas


def canvas_conv(current_canvas, final_canvas):
    current_canvas = current_canvas.data
    final_canvas = final_canvas.data
    canvas_encode = torch.zeros((final_canvas.size(0), 25, 6)).fill_(-1).cuda()
    canvas_encode[:, :, :4] = final_canvas  # copy final canvas
    final_canvas_mask = (final_canvas.sum(dim=2) >= 0)
    canvas_encode[:, :, 4][final_canvas_mask] = 1
    current_canvas_mask = (current_canvas.sum(dim=2) >= 0)
    canvas_encode[:, :, 4][current_canvas_mask] = 0
    conv_feats = torch.zeros(canvas_encode.size(0), 25).cuda()  # Bx25
    conv_feats[current_canvas_mask] = 1
    conv_feats = conv_feats.view(canvas_encode.size(0), 1, 5, 5)
    conv_feats = F.pad(conv_feats, pad=(1,1,1,1), value=0)
    filters = Variable(torch.ones(1, 1, 3, 3).cuda())
    filters.data[:, :, 1, 1] = 0
    conv_feats = F.conv2d(conv_feats, filters)
    canvas_encode[:, :, 5] = conv_feats.data.view(conv_feats.size(0), 25)
    return Variable(canvas_encode)


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
        rewards = Variable(output.view(batch_size, -1).data)
        model.rewards = rewards
        # model.rewards = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-5)
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


def attend(memory, h, trans_fn):
    att = torch.cat([h, memory], 2)  # Bx25x68
    att = trans_fn(att).squeeze()  # Bx25
    weight = F.softmax(att, dim=1)  # Bx25
    att = torch.bmm(weight.unsqueeze(1), memory).squeeze(1)  # Bx4
    return att


def hard_attend(memory, h, trans_fn):
    att = torch.cat([h, memory], 2)  # Bx25x68
    att = trans_fn(att).squeeze()  # Bx25
    weight = F.softmax(att, dim=1)  # Bx25
    m = Categorical(weight)
    action = m.sample()
    model.saved_log_probs.append(m.log_prob(action))
    # TODO use gather?
    output = memory.data.new(memory.size(0), memory.size(2))  # Bx6
    action = action.data
    for i in range(output.size(0)):
        output[i] = memory[i, action[i]].data
    return Variable(output)


def hard_attend2(memory, h, trans_fn, target_obj, is_training):
    if is_training:
        return target_obj.float()
    mem_data = memory.data
    mask = (mem_data[:,:,4] >= 1) & (mem_data[:, :, 5] >= 1) # Bx25
    output = memory.data.new(memory.size(0), memory.size(2))  # Bx6
    for i in range(output.size(0)):
        output[i] = memory[i, torch.nonzero(mask[i])[0, 0]].data
    return Variable(output)

    # mask = mask.float() + 1e-5
    # probs = mask / mask.sum(dim=1, keepdim=True)
    # m = Categorical(probs)
    # action = m.sample()
    # output = memory.data.new(memory.size(0), memory.size(2))  # Bx6
    # for i in range(output.size(0)):
    #     output[i] = memory[i, action[i]].data
    # return Variable(output)


def dual_attend(h, memory1, memory2, trans_fn1, trans_fn2, trans_fn3, extra):
    target_obj, ref_obj, is_training = extra
    h2 = torch.unsqueeze(h, 1)  # Bx1x64
    h2 = h2.repeat(1, 25, 1)  # Bx25x64
    att1 = hard_attend2(memory1, h2, trans_fn1, target_obj, is_training)
    # att1 = attend(memory1, h2, trans_fn1)
    att2 = attend(memory2, h2, trans_fn2)
    att3 = attend(memory2, h2, trans_fn3)
    # att = torch.cat([att1, att2], dim=1)  # Bx8
    # att = torch.cat([target_obj.float(), att2, att3], dim=1)  # Bx8
    att = torch.cat([att1[:, :4], att2, att3], dim=1)  # Bx8
    return att


class InstAtt(nn.Module):
    def __init__(self, vocab_size, max_seq_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.rnn_size = 128
        self.att_hidden_size = 64
        self.input_size = 64
        # https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/models/AttModel.py#L43
        self.embed = nn.Embedding(vocab_size + 1, self.input_size)
        # https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/models/AttModel.py#L372
        self.att_lstm_cell = nn.LSTMCell(self.input_size + self.rnn_size, self.rnn_size)
        self.lang_lstm_cell = nn.LSTMCell(self.rnn_size * 2, self.rnn_size)
        self.fc_out = nn.Linear(self.rnn_size, self.vocab_size + 1)
        # self.fc_att1 = nn.Sequential(nn.Linear(self.rnn_size + 6, self.att_hidden_size), nn.ReLU(), nn.Linear(self.att_hidden_size, 1))
        self.fc_att1 = None
        self.fc_att2 = nn.Sequential(nn.Linear(self.rnn_size + 4, self.att_hidden_size), nn.ReLU(), nn.Linear(self.att_hidden_size, 1))
        self.fc_att3 = nn.Sequential(nn.Linear(self.rnn_size + 4, self.att_hidden_size), nn.ReLU(), nn.Linear(self.att_hidden_size, 1))
        self.att_trans = nn.Sequential(nn.Linear(4+4+4, self.att_hidden_size), nn.ReLU(), nn.Linear(self.att_hidden_size, self.rnn_size))
        self.saved_log_probs = []
        self.rewards = None

    def forward(self, inst, prev_canvas, final_canvas, extra=None):
        # http://pytorch.org/docs/master/nn.html#torch.nn.LSTMCell
        conv_canvas = canvas_conv(prev_canvas, final_canvas)  # Bx5x7x7
        # memory = torch.cat([diff_canvas, prev_canvas], dim=1) # 50x4
        batch_size = inst.size(0)
        state = (Variable(torch.zeros(2, batch_size, self.rnn_size).cuda()),
                 Variable(torch.zeros(2, batch_size, self.rnn_size).cuda()))
        outputs = []
        # inst: [0, a, b, c, 0]
        for i in range(inst.size(1) - 1):
            if i >= 1 and inst[:, i].data.sum() == 0:
                break
            step_output, state = self.step(inst[:, i], state, conv_canvas, prev_canvas, extra)
            outputs.append(step_output)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)  # Bx(seq_len)x(vocab_size+1)

    def step(self, step_input, state, diff_canvas, prev_canvas, extra=None):
        h_lang, c_lang = state[0][1], state[1][1]
        h_att, c_att = (state[0][0], state[1][0])
        xt = self.embed(step_input)
        att_lstm_input = torch.cat([h_lang, xt], 1)
        h_att, c_att = self.att_lstm_cell(att_lstm_input, (h_att, c_att))
        att = dual_attend(h_att, diff_canvas, prev_canvas, self.fc_att1, self.fc_att2, self.fc_att3, extra)
        att = self.att_trans(att)  # Bx64
        lang_lstm_input = torch.cat([att, h_att], 1)
        h_lang, c_lang = self.lang_lstm_cell(lang_lstm_input, (h_lang, c_lang))
        step_output = F.log_softmax(self.fc_out(h_lang), dim=1)  # Bx(vocab_size+1)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))
        return step_output, state

    def sample(self, prev_canvas, final_canvas, extra=None):
        assert not self.embed.training
        assert not self.att_lstm_cell.training
        assert not self.fc_out.training
        sample_max = True
        temperature = 0.8
        seq = []
        conv_canvas = canvas_conv(prev_canvas, final_canvas)
        batch_size = prev_canvas.size(0)
        state = (Variable(torch.zeros(2, batch_size, self.rnn_size).cuda()),
                 Variable(torch.zeros(2, batch_size, self.rnn_size).cuda()))
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
            logprobs, state = self.step(Variable(it, volatile=True), state, conv_canvas, prev_canvas, extra)
        return torch.t(torch.stack(seq)).cpu()

# sample_loader = get_shape2d_loader(split='sample', batch_size=2)
# print(len(sample_loader.dataset))


train_loader = get_shape2d_loader(split='train', batch_size=args.batch_size)
val_loader = get_shape2d_loader(split='val', batch_size=50)
assert train_loader.dataset.vocab_size == val_loader.dataset.vocab_size
assert train_loader.dataset.max_seq_length == val_loader.dataset.max_seq_length
# database = [d['current_instruction'] for d in train_loader.dataset.data]
# matched = []
# for d in tqdm(val_loader.dataset.data):
#     if d['current_instruction'] in database:
#         matched.append(database.index(d['current_instruction']))
# print(len(matched))
# dd = val_loader.dataset.data[0]
# print(dd['next_object'])
# sample = dd['current_instruction']
# for i, d in enumerate(database):
#     if d == sample:
#         if(train_loader.dataset.data[i]['next_object'] == dd['next_object'] and train_loader.dataset.data[i]['ref_obj'] == dd['ref_obj']):
#             print(i)

val_loader_iter = iter(val_loader)
# sample_loader = get_shape2d_loader(split='sample', batch_size=2)
# print(len(sample_loader.dataset))

model = InstAtt(train_loader.dataset.vocab_size, train_loader.dataset.max_seq_length)
# model.load_state_dict(torch.load('models_topdown/model_20.pth'))
# model.load_state_dict(torch.load('models_topdown_3att_att64_hardatt/model_13.pth'))
model.cuda()
loss_fn = LanguageModelCriterion()


# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)


def eval_sample():
    model.eval()
    val_data = next(val_loader_iter)
    val_raw_data = val_data[-1]
    data = [Variable(_, volatile=True).cuda() for _ in val_data[:-1]]
    prev_canvas, inst, next_obj, final_canvas, ref_obj = data
    diff_canvas = compute_diff_canvas(prev_canvas, final_canvas)
    diff_canvas_objs = [train_loader.dataset.decode_canvas(diff_canvas.data[i]) for i in range(diff_canvas.size(0))]
    samples = model.sample(prev_canvas, final_canvas, (next_obj, ref_obj, False))
    samples = decode_sequence(train_loader.dataset.ix_to_word, samples)
    for i in range(prev_canvas.size(0)):
        val_raw_data[i]['predicted_instruction'] = samples[i]
    for i, d in enumerate(val_raw_data):
        d['prev_canvas'] = render_canvas(d['prev_canvas']).replace(' ', '')
        d['final_canvas'] = render_canvas(d['final_canvas']).replace(' ', '')
        d['diff_canvas'] = render_canvas(diff_canvas_objs[i]).replace(' ', '')
    r = requests.post("http://vision.cs.virginia.edu:5001/new_task", json=val_raw_data)
    print('request sent')


def train(epoch):
    model.train()
    adjust_lr(optimizer, epoch, args.lr, decay_rate=0.2)
    for batch_idx, data in enumerate(train_loader):
        raw_data = data[-1]
        data = [Variable(_, requires_grad=False).cuda() for _ in data[:-1]]
        prev_canvas, inst, next_obj, final_canvas, ref_obj = data
        optimizer.zero_grad()
        loss = loss_fn(model(inst, prev_canvas, final_canvas, (next_obj, ref_obj, True)), inst[:, 1:])
        policy_loss = 0
        for i in range(len(model.saved_log_probs)):
            policy_loss += (-model.saved_log_probs[i] * model.rewards[:, i]).sum()
        (loss + policy_loss).backward()
        clip_gradient(optimizer, 0.1)
        optimizer.step()
        del model.saved_log_probs[:]
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f})'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))
    # torch.save(model.state_dict(), 'models_topdown_3att_att64_hardatt/model_{}.pth'.format(epoch))
    # torch.save(optimizer.state_dict(), 'models_topdown_3att_att64_hardatt/optimizer_{}.pth'.format(epoch))
    torch.save(model.state_dict(), 'models_topdown_3att_att64_hardatt_diff/model_{}.pth'.format(epoch))
    torch.save(optimizer.state_dict(), 'models_topdown_3att_att64_hardatt_diff/optimizer_{}.pth'.format(epoch))


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        eval_sample()
        del model.saved_log_probs[:]
        train(epoch)


