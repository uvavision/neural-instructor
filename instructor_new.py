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


def canvas_conv(prev_canvas, final_canvas):
    prev_canvas = prev_canvas.data
    final_canvas = final_canvas.data
    canvas_encode = torch.zeros((final_canvas.size(0), 25, 7)).fill_(-1).cuda()
    canvas_encode[:, :, :4] = final_canvas  # copy final canvas
    final_canvas_mask = (final_canvas.sum(dim=2) >= 0)
    canvas_encode[:, :, 4][final_canvas_mask] = 1
    prev_canvas_mask = (prev_canvas.sum(dim=2) >= 0)
    canvas_encode[:, :, 4][prev_canvas_mask] = 0
    # objects in the prev_canvas but disagree with final_canvas
    disagree_mask = prev_canvas_mask & ((prev_canvas == final_canvas).sum(dim=2) != 4)
    disagrees = torch.nonzero(disagree_mask)
    for i in range(disagrees.size(0)):
        d = disagrees[i]
        canvas_encode[d[0], d[1], :4] = prev_canvas[d[0], d[1]]
        # print("{} {}: on conv canvas: {}".format(d[0], d[1], list(canvas_encode[d[0], d[1]])))
        # print("{} {}: on prev canvas: {}".format(d[0], d[1], list(prev_canvas[d[0], d[1]])))
        # print("{} {}: on final canvas: {}".format(d[0], d[1], list(final_canvas[d[0], d[1]])))
    canvas_encode[:, :, 5][disagree_mask] = 1
    conv_feats = torch.zeros(canvas_encode.size(0), 25).cuda()  # Bx25
    conv_feats[prev_canvas_mask] = 1
    conv_feats = conv_feats.view(canvas_encode.size(0), 1, 5, 5)
    conv_feats = F.pad(conv_feats, pad=(1,1,1,1), value=0)
    filters = Variable(torch.ones(1, 1, 3, 3).cuda())
    filters.data[:, :, 1, 1] = 0
    conv_feats = F.conv2d(conv_feats, filters)
    canvas_encode[:, :, 6] = conv_feats.data.view(conv_feats.size(0), 25)
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
        # reward is the log likelihood of the target sequence
        rewards = (input.gather(1, target) * mask).view(batch_size, -1).sum(dim=1).data
        # TODO use mean as baseline
        model.rewards = Variable(rewards - model.running_baseline)
        model.running_baseline = 0.9 * model.running_baseline + 0.1 * rewards.mean()
        # model.rewards = Variable((rewards - rewards.mean()) / (rewards.std() + 1e-6))
        output = torch.sum(output) / torch.sum(mask)

        return output


def attend(memory, h, trans_fn):
    att = torch.cat([h, memory], 2)  # Bx25x68
    att = trans_fn(att).squeeze()  # Bx25
    weight = F.softmax(att, dim=1)  # Bx25
    att = torch.bmm(weight.unsqueeze(1), memory).squeeze(1)  # Bx4
    return att


def attend_target(conv_canvas, trans_fn):
    # conv_canvas: Bx25x6
    att = trans_fn(conv_canvas).squeeze()  # Bx25
    weight = F.softmax(att, dim=1)  # Bx25
    att = torch.bmm(weight.unsqueeze(1), conv_canvas).squeeze(1)  # Bx6
    return att


standard_loc0 = torch.LongTensor([[0]]).repeat(1000, 25).cuda()
standard_loc2 = torch.LongTensor([[2]]).repeat(1000, 25).cuda()
standard_loc4 = torch.LongTensor([[4]]).repeat(1000, 25).cuda()


def get_actionable_obj_mask(conv_canvas):
    batch_size = conv_canvas.size(0)
    assert batch_size <= standard_loc0.size(0)
    data = conv_canvas.data.long()
    mask1 = data[:, :, 4] >= 1  # object in final canvas but not in current canvas
    mask2 = data[:, :, 5] >= 1  # Bx25 surrounded by objects in the current canvas
    loc0, loc2, loc4 = standard_loc0[:batch_size], standard_loc2[:batch_size], standard_loc4[:batch_size]
    mask00 = torch.eq(data[:, :, 3], loc0) | torch.eq(data[:, :, 3], loc4)  # (, 0) or (, 4)
    mask01 = torch.eq(data[:, :, 2], loc0) & mask00  # (0, 0) or (0, 4)
    mask02 = torch.eq(data[:, :, 2], loc4) & mask00  # (4, 0) or (4, 4)
    mask03 = torch.eq(data[:, :, 2], loc2) & torch.eq(data[:, :, 3], loc2)  # (2, 2)
    mask = mask1 & (mask2 | mask01 | mask02 | mask03)
    return mask


def hard_attend3(memory, h, trans_fn, target_obj, is_training):
    if is_training:
        return target_obj.float()
    mask = get_actionable_obj_mask(memory)
    output = memory.data.new(memory.size(0), memory.size(2))  # Bx6
    for i in range(output.size(0)):
        output[i] = memory[i, torch.nonzero(mask[i])[0, 0]].data
    return Variable(output.float())


def hard_attend_reinforce_gt(conv_canvas, h, trans_fn, target_obj, is_training):
    # FIXME
    return target_obj.float()
    if model.training:
        return target_obj.float()
    # conv_canvas: Bx25x7
    if model.sampled_actions is None:
        model.sampled_actions = []
        data = conv_canvas.data.long()
        mask = (data[:, :, 4] == 1) | (data[:, :, 5] == 1)
        for i in range(conv_canvas.size(0)):
            model.sampled_actions.append(random.choice(list(torch.nonzero(mask[i]).squeeze())))
    output = conv_canvas.data.new(conv_canvas.size(0), conv_canvas.size(2))  # Bx7
    for i in range(output.size(0)):
        output[i] = conv_canvas.data[i, model.sampled_actions[i]]
    return Variable(output)

def hard_attend_reinforce(memory, h, trans_fn, target_obj, is_training):
    actions = model.sampled_actions
    if actions is None:
        att = trans_fn(memory).squeeze()  # Bx25
        weight = F.softmax(att, dim=1)  # Bx25
        m = Categorical(weight)
        actions = m.sample()
        model.saved_log_probs = m.log_prob(actions)
        model.sampled_actions = actions
    # TODO use gather?
    output = memory.data.new(memory.size(0), memory.size(2))  # Bx6
    for i in range(output.size(0)):
        output[i] = memory[i, actions.data[i]].data
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


def dual_attend(h, conv_canvas, prev_canvas, trans_fn1, trans_fn2, trans_fn3, extra):
    target_obj, ref_obj, is_training = extra
    h2 = torch.unsqueeze(h, 1)  # Bx1x64
    h2 = h2.repeat(1, 25, 1)  # Bx25x64
    att1 = hard_attend_reinforce_gt(conv_canvas, h2, trans_fn1, target_obj, is_training)
    # att1 = attend(memory1, h2, trans_fn1)
    # att1 = attend_target(memory1, trans_fn1)
    att2 = attend(prev_canvas, h2, trans_fn2)
    att3 = attend(prev_canvas, h2, trans_fn3)
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
        # self.fc_att1 = nn.Sequential(nn.Linear(7, 3), nn.ReLU(),nn.Linear(3, 1))
        self.fc_att1 = None
        self.fc_att2 = nn.Sequential(nn.Linear(self.rnn_size + 4, self.att_hidden_size), nn.ReLU(), nn.Linear(self.att_hidden_size, 1))
        self.fc_att3 = nn.Sequential(nn.Linear(self.rnn_size + 4, self.att_hidden_size), nn.ReLU(), nn.Linear(self.att_hidden_size, 1))
        self.att_trans = nn.Sequential(nn.Linear(4+4+4, self.att_hidden_size), nn.ReLU(), nn.Linear(self.att_hidden_size, self.rnn_size))
        self.saved_log_probs = None
        self.rewards = None
        self.sampled_actions = None
        self.running_baseline = 0

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

    def step(self, step_input, state, conv_canvas, prev_canvas, extra=None):
        h_lang, c_lang = state[0][1], state[1][1]
        h_att, c_att = (state[0][0], state[1][0])
        xt = self.embed(step_input)
        att_lstm_input = torch.cat([h_lang, xt], 1)
        h_att, c_att = self.att_lstm_cell(att_lstm_input, (h_att, c_att))
        att = dual_attend(h_att, conv_canvas, prev_canvas, self.fc_att1, self.fc_att2, self.fc_att3, extra)
        att = self.att_trans(att)  # Bx64
        lang_lstm_input = torch.cat([att, h_att], 1)
        h_lang, c_lang = self.lang_lstm_cell(lang_lstm_input, (h_lang, c_lang))
        step_output = F.log_softmax(self.fc_out(h_lang), dim=1)  # Bx(vocab_size+1)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))
        return step_output, state

    def sample(self, prev_canvas, final_canvas, extra=None):
        # assert not self.embed.training
        # assert not self.att_lstm_cell.training
        # assert not self.fc_out.training
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
            # logprobs, state = self.step(Variable(it, volatile=True), state, conv_canvas, prev_canvas, extra)
            logprobs, state = self.step(Variable(it), state, conv_canvas, prev_canvas, extra)
        return torch.t(torch.stack(seq)).cpu()

# sample_loader = get_shape2d_loader(split='sample', batch_size=2)
# print(len(sample_loader.dataset))


val_loader = get_shape2d_instructor_data_loader(split='val', batch_size=20, shuffle=False)
train_loader = get_shape2d_instructor_data_loader(split='val', batch_size=args.batch_size)
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
# model.load_state_dict(torch.load('models_topdown_3att_att64_content_planning_absmix_reinforce/model_5.pth'))
# model.load_state_dict(torch.load('models_topdown_3att_att64_content_planning_absmix_reinforce_running_baseline/model_20.pth'))
# model.load_state_dict(torch.load("instructor_trained_with_painter/model_20.pth"))
# model.load_state_dict(torch.load("instructor_model_new//model_20.pth"))
model.load_state_dict(torch.load("instructor_model_new_retrain//model_20.pth"))
model.cuda()
loss_fn = LanguageModelCriterion()

painter_model = Shape2DPainterNet(train_loader.dataset.vocab_size)
painter_model.load_state_dict(torch.load('painter_model_new_add_remove///model_20.pth'))
painter_model.cuda()
painter_model.eval()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
# fc_att1_optimizer = optim.Adam(model.fc_att1.parameters(), lr=1e-3, weight_decay=0)


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
        val_raw_data[i]['predicted_instruction'] = '{} - {} ({} {} {} {} -> {} {} {} {}) [multi-memory]'.format(
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


def get_painter_rewards(target_obj_prediction, final_canvas):
    batch_size = target_obj_prediction.size(0)
    rewards = torch.zeros(batch_size)
    for i in range(batch_size):
        pos = target_obj_prediction[i, 2] * 5 + target_obj_prediction[i, 3]
        if pos < 25 and torch.equal(target_obj_prediction[i], final_canvas[i, pos]):
            rewards[i] = 1
        else:
            rewards[i] = -1
    return rewards


def train_with_painter(painter_model, epoch):
    model.train()
    painter_model.eval()
    epoch_rewards = []
    for batch_idx, data in enumerate(train_loader):
        raw_data = data[-1]
        data = [Variable(_, requires_grad=False).cuda() for _ in data[:-1]]
        prev_canvas, inst, next_obj, final_canvas, ref_obj = data
        fc_att1_optimizer.zero_grad()
        samples = model.sample(prev_canvas, final_canvas, (next_obj, ref_obj, None))
        target_obj_prediction = get_painter_model_prediction(painter_model, samples, prev_canvas)
        # print((torch.eq(target_obj_predict, next_obj.data).sum(dim=1) == 4).long().sum())
        painter_rewards = get_painter_rewards(target_obj_prediction, final_canvas.data.long())

        # samples2 = decode_sequence(train_loader.dataset.ix_to_word, samples)
        # diff_canvas = compute_diff_canvas(prev_canvas, final_canvas)
        # diff_canvas_objs = [train_loader.dataset.decode_canvas(diff_canvas.data[i]) for i in range(diff_canvas.size(0))]
        # for i in range(prev_canvas.size(0)):
        #     raw_data[i]['predicted_instruction'] = samples2[i]
        # for i, d in enumerate(raw_data):
        #     d['prev_canvas'] = render_canvas(d['prev_canvas']).replace(' ', '')
        #     d['final_canvas'] = render_canvas(d['final_canvas']).replace(' ', '')
        #     d['diff_canvas'] = render_canvas(diff_canvas_objs[i]).replace(' ', '')
        # r = requests.post("http://vision.cs.virginia.edu:5001/new_task", json=raw_data)
        # print('request sent')

        policy_loss = (-model.saved_log_probs * Variable(painter_rewards.cuda())).sum()
        policy_loss.backward()
        clip_gradient(fc_att1_optimizer, 0.1)
        fc_att1_optimizer.step()
        model.saved_log_probs = None
        model.sampled_actions = None
        model.rewards = None

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tRewards: {:.6f})'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), painter_rewards.mean()))
        epoch_rewards.append(painter_rewards.mean())
    print("epoch reward {}".format(np.array(epoch_rewards).mean()))
    torch.save(model.state_dict(), 'instructor_trained_with_painter/model_{}.pth'.format(epoch))


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
        # policy_loss = (-model.saved_log_probs * model.rewards).sum()
        # (loss + policy_loss).backward()
        clip_gradient(optimizer, 0.1)
        optimizer.step()
        model.saved_log_probs = None
        model.sampled_actions = None
        model.rewards = None
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f})'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))
    # torch.save(model.state_dict(), 'models_topdown_3att_att64_hardatt/model_{}.pth'.format(epoch))
    # torch.save(optimizer.state_dict(), 'models_topdown_3att_att64_hardatt/optimizer_{}.pth'.format(epoch))
    # torch.save(model.state_dict(), 'instructor_model_new/model_{}.pth'.format(epoch))
    # torch.save(optimizer.state_dict(), 'models_topdown_3att_att64_content_planning_absmix_reinforce_running_baseline/optimizer_{}.pth'.format(epoch))
    torch.save(model.state_dict(), 'instructor_model_new_retrain/model_{}.pth'.format(epoch))


if __name__ == '__main__':
    eval_sample()
    for epoch in range(1, args.epochs + 1):
    #     # train_with_painter(painter_model, epoch)
    #     # eval_sample()
        train(epoch)
        model.saved_log_probs = None
        model.sampled_actions = None
        model.rewards = None



