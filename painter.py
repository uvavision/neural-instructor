from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from Shape2D import get_shape2d_loader
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


class InstEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.rnn_size = 64
        self.input_size = 64
        # https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/models/AttModel.py#L43
        self.embed = nn.Embedding(vocab_size + 1, self.input_size)
        # https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/models/AttModel.py#L372
        self.lstm_cell = nn.LSTMCell(self.input_size, self.rnn_size)

    def forward(self, inst):
        # http://pytorch.org/docs/master/nn.html#torch.nn.LSTMCell
        batch_size = inst.size(0)
        h = Variable(torch.zeros(batch_size, self.rnn_size).cuda())
        c = Variable(torch.zeros(batch_size, self.rnn_size).cuda())
        for i in range(inst.size(1)):
            input = self.embed(inst[:, i])
            h, c = self.lstm_cell(input, (h, c))
        # only want the last hidden state
        return h


class CanvasEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_size = 4
        # self.obj_embed = nn.Linear(3, self.embed_size)

    def forward(self, canvas):
        # canvas: batch_size x 25 x 3
        # obj_embedding = self.obj_embed(canvas)
        # result = canvas[:, :, 2:]
        result = canvas
        assert result.size(2) == self.embed_size
        return result
        # return obj_embedding
        # canvas_embedding = obj_embedding.sum(1)  # batch x 64
        # return canvas_embedding


class Shape2DPainterNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.inst_encoder = InstEncoder(vocab_size)
        self.use_mask = True
        # self.canvas_encoder = CanvasEncoder()
        self.hidden_size = 64
        self.fc_color = nn.Linear(self.inst_encoder.rnn_size, 3)
        self.fc_shape = nn.Linear(self.inst_encoder.rnn_size, 3)
        self.fc_offset = nn.Linear(self.inst_encoder.rnn_size, 2)
        # self.canvas_embed = nn.Linear(4, 64)
        # self.fc_ref_obj = nn.Linear(64, 1)
        # self.fc_ref_obj = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.fc_ref_obj = nn.Sequential(nn.Linear(69, 32), nn.ReLU(), nn.Linear(32, 1))
        # self.fc_dir = nn.Linear(self.inst_encoder.rnn_size, 8)
        # self.fc_row = nn.Linear(2 + self.canvas_encoder.embed_size, 1, bias=False)
        # self.fc_col = nn.Linear(2 + self.canvas_encoder.embed_size, 1, bias=False)
        # self.fc_row = nn.Linear(2, 1, bias=False)
        # self.fc_col = nn.Linear(2, 1, bias=False)
        # self.fc1 = nn.Linear(2 + self.canvas_encoder.embed_size, self.hidden_size)

    def forward(self, inst, prev_canvas, ref_obj, target_obj):
        inst_embedding = self.inst_encoder(inst) # Bx64
        act_tag = torch.zeros(inst_embedding.size(0))
        inst_data = inst.data
        for i in range(act_tag.size(0)):
            inst_data = inst[i].data
            if inst_data[inst_data>=1][-1] == 9:
                act_tag[i] = 1
        act_tag = Variable(act_tag.cuda())
        inst2 = torch.cat([inst_embedding, act_tag.unsqueeze(1)], 1)
        inst2 = torch.unsqueeze(inst2, 1)  # Bx1x64
        inst2 = inst2.repeat(1, 25, 1) # Bx25x64
        # canvas_embedding = self.canvas_encoder(prev_canvas) # Bx25x16
        ref_obj_e = torch.cat([inst2, prev_canvas], 2) # Bx25x68
        # canvas_embedding = self.canvas_embed(prev_canvas)
        # ref_obj_e = canvas_embedding + inst2
        ref_obj_e = self.fc_ref_obj(ref_obj_e).squeeze() # Bx25
        weight = F.softmax(ref_obj_e, dim=1)
        ref_obj_predict = torch.bmm(weight.unsqueeze(1), prev_canvas).squeeze(1)
        # ref_obj_out = F.log_softmax(ref_obj_e, dim=1)

        # if self.use_mask:
        #     mask = torch.zeros(canvas_embedding.size())
        #     for i in range(ref_obj.data.size(0)):
        #         # assert ref_obj[i].sum() > 0
        #         mask[i][ref_obj.data[i][-2]*5+ref_obj.data[i][-1]].fill_(1)
        #     mask = Variable(mask.cuda())
        #     canvas_embedding = (canvas_embedding * mask).sum(dim=1)
        # print(torch.abs(canvas_embedding.data - ref_obj.float().cuda()).sum())
        offset_embedding = self.fc_offset(inst_embedding)
        # offset_embedding = Variable((target_obj.data[:, 2:] - ref_obj[:, 2:].cuda()).float())
        # offset_embedding =
        offset_embedding = F.hardtanh(offset_embedding)
        # dir_embedding = F.softmax(self.fc_dir(inst_embedding))
        # embedding = torch.cat([offset_embedding, canvas_embedding], 1)  # Bx25x(64+16)
        # embedding = offset_embedding + canvas_embedding

        # global train_loader
        # ref_obj_target = ref_obj[:, -2] * 5 + ref_obj[:, -1]
        # for ix in range(prev_canvas.size(0)):
        #     inst_str = ' '.join(map(train_loader.dataset.ix_to_word.get, list(inst[ix].data)))
        #     print(inst_str)
        #     ref_obj = prev_canvas.data[ix, ref_obj_target.data[ix]]
        #     ref_obj_color = train_loader.dataset.num2color[ref_obj[0]]
        #     ref_obj_shape = train_loader.dataset.num2shape[ref_obj[1]]
        #     print("ref obj: {} {} {} {}".format(ref_obj_color, ref_obj_shape, ref_obj[2], ref_obj[3]))
        #     print(inst[ix].data)
        #     print("\n")
            # print("{} {}".format(canvas_embedding.data[ix][-2], canvas_embedding.data[ix][-1]))
            # print("{} {}".format(target_obj.data[ix][-2], target_obj.data[ix][-1]))


        # mask = torch.zeros(embedding.size())
        # for i in range(ref_obj.size(0)):
        #     if ref_obj[i].sum() > 0:
        #         mask[i][ref_obj[i][-1]].fill_(1)
        # mask = Variable(mask.cuda())
        # embedding = embedding * mask
        # embedding = torch.cat([inst_embedding, canvas_embedding], 1)
        # x = F.relu(self.fc1(embedding)) # Bx25x32
        # # mask based on ref_obj
        # mask = torch.zeros(x.size())
        # for i in range(ref_obj.size(0)):
        #     if ref_obj[i].sum() > 0:
        #         mask[i][ref_obj[i][-1]].fill_(1)
        # mask = Variable(mask.cuda())
        # x = x * mask
        # x = F.dropout(x, training=self.training)
        # if self.use_canvas_data:
        #     x = x.sum(dim=1) # Bx32
        color_out = F.log_softmax(self.fc_color(inst_embedding), dim=1)
        shape_out = F.log_softmax(self.fc_shape(inst_embedding), dim=1)
        # row_out = F.log_softmax(self.fc_row(embedding), dim=1)
        # col_out = F.log_softmax(self.fc_col(embedding), dim=1)
        # row_out = self.fc_row(embedding)
        # col_out = self.fc_col(embedding)
        # row_offset_out = offset_embedding[:, 0]
        # col_offset_out = offset_embedding[:, 1]
        row_out = ref_obj_predict[:, 2] + offset_embedding[:, 0]
        col_out = ref_obj_predict[:, 3] + offset_embedding[:, 1]
        return color_out, shape_out, row_out, col_out


class Shape2DObjCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_out, target_obj, ref_obj):
        color_target = target_obj[:, 0]
        shape_target = target_obj[:, 1]
        color_out, shape_out, row_out, col_out = model_out
        loss = F.nll_loss(color_out, color_target) + \
               F.nll_loss(shape_out, shape_target) + \
               F.l1_loss(row_out, target_obj[:, 2].float()) + \
               F.l1_loss(col_out, target_obj[:, 3].float())
        return loss


# https://discuss.pytorch.org/t/adaptive-learning-rate/320/2
def adjust_lr(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.8 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def compute_accuracy(model_out, target_obj, ref_obj):
    color_target = target_obj[:, 0]
    shape_target = target_obj[:, 1]
    color_out, shape_out, row_out, col_out = model_out
    # ref_obj_target = ref_obj[:, -2] * 5 + ref_obj[:, -1]
    batch_size = target_obj.size(0)
    color_accuracy = torch.eq(torch.max(color_out.data, dim=1)[1], color_target.data).sum()/batch_size
    shape_accuracy = torch.eq(torch.max(shape_out.data, dim=1)[1], shape_target.data).sum()/batch_size
    # row_accuracy = torch.eq(torch.max(row_out, dim=1)[1], row_target).sum() / batch_size
    # col_accuracy = torch.eq(torch.max(col_out, dim=1)[1], col_target).sum() / batch_size
    # row_accuracy = torch.abs(row_out - row_target).mean()
    # col_accuracy = torch.abs(col_out - col_target).mean()
    # row_offset = target_obj[:, 2] - ref_obj[:, 2]
    # col_offset = target_obj[:, 3] - ref_obj[:, 3]
    row_accuracy = F.l1_loss(row_out, target_obj[:, 2].float())
    col_accuracy = F.l1_loss(col_out, target_obj[:, 3].float())
    # TODO
    # row2 = torch.abs(row_offset_out.data - row_offset.data.float()).mean()
    # col2 = torch.abs(col_offset_out.data - col_offset.data.float()).mean()
    return color_accuracy, shape_accuracy, row_accuracy.data[0], col_accuracy.data[0]


train_loader = get_shape2d_loader(split='train', batch_size=args.batch_size)
test_loader = get_shape2d_loader(split='val', batch_size=args.batch_size)
assert train_loader.dataset.vocab_size == test_loader.dataset.vocab_size
assert train_loader.dataset.max_seq_length == test_loader.dataset.max_seq_length

model = Shape2DPainterNet(train_loader.dataset.vocab_size)
model.cuda()
loss_fn = Shape2DObjCriterion()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)


def train(epoch):
    # adjust_lr(optimizer, epoch)
    model.train()
    for batch_idx, (prev_canvas, inst, next_obj, final_canvas, ref_obj) in enumerate(train_loader):
        # for ix in range(args.batch_size):
        #     # ix = random.randint(0, args.batch_size-1)
        #     inst_str = ' '.join(map(train_loader.dataset.ix_to_word.get, list(inst[ix])))
        #     next_obj_color = train_loader.dataset.num2color[next_obj[ix][0]]
        #     next_obj_shape = train_loader.dataset.num2shape[next_obj[ix][1]]
        #     print(inst_str)
        #     print("target obj: {} {} {} {}".format(next_obj_color, next_obj_shape, next_obj[ix][2], next_obj[ix][3]))
        #     if ref_obj[ix].sum() > 0:
        #         ref_obj_color = train_loader.dataset.num2color[ref_obj[ix][0]]
        #         ref_obj_shape = train_loader.dataset.num2shape[ref_obj[ix][1]]
        #         print("ref obj: {} {} {} {}".format(ref_obj_color, ref_obj_shape, ref_obj[ix][2], ref_obj[ix][3]))
        #     print("\n")
        prev_canvas = Variable(prev_canvas.cuda())
        inst = Variable(inst.cuda())
        next_obj = Variable(next_obj.cuda())
        ref_obj = Variable(ref_obj.cuda())
        output = model(inst, prev_canvas, ref_obj, next_obj)
        loss = loss_fn(output, next_obj, ref_obj)
        loss.backward()
        clip_gradient(optimizer, 0.1)
        optimizer.step()
        color_accuracy, shape_accuracy, row_accuracy, col_accuray = compute_accuracy(output, next_obj, ref_obj)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} ({:.3f} {:.3f} {} {})'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0], color_accuracy, shape_accuracy, row_accuracy, col_accuray))


def model_test():
    model.eval()
    test_loss = 0
    correct = 0
    for prev_canvas, inst, next_obj, final_canvas in test_loader:
        prev_canvas = Variable(prev_canvas.cuda(), volatile=True)
        inst = Variable(inst.cuda(), volatile=True)
        next_obj = Variable(next_obj.cuda(), volatile=True)
        output = model(inst, prev_canvas)
        loss = loss_fn(output, next_obj)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    # model_test()
