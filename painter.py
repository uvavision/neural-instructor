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
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
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
        self.embed_size = 8
        self.obj_embed = nn.Linear(3, self.embed_size)

    def forward(self, canvas):
        # canvas: batch_size x 25 x 3
        obj_embedding = self.obj_embed(canvas)
        return obj_embedding
        # canvas_embedding = obj_embedding.sum(1)  # batch x 64
        # return canvas_embedding


class Shape2DPainterNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.inst_encoder = InstEncoder(vocab_size)
        self.canvas_encoder = CanvasEncoder()
        self.hidden_size = 32
        self.fc1 = nn.Linear(self.inst_encoder.rnn_size + self.canvas_encoder.embed_size, 32)
        # self.fc1 = nn.Linear(self.inst_encoder.rnn_size, 32)
        self.fc_color = nn.Linear(32, 3)
        self.fc_shape = nn.Linear(32, 3)
        self.fc_loc = nn.Linear(32, 25)

    def forward(self, inst, prev_canvas):
        inst_embedding = self.inst_encoder(inst) # Bx64
        inst_embedding = torch.unsqueeze(inst_embedding, 1) # Bx1x64
        inst_embedding = inst_embedding.repeat(1, 25, 1) # Bx25x64
        canvas_embedding = self.canvas_encoder(prev_canvas) # Bx25x16
        embedding = torch.cat([inst_embedding, canvas_embedding], 2) # Bx25x(64+16)
        # embedding = torch.cat([inst_embedding, canvas_embedding], 1)
        x = F.relu(self.fc1(embedding)) # Bx25x32
        x = F.dropout(x, training=self.training)
        x = x.sum(dim=1) # Bx32
        color_out = F.log_softmax(self.fc_color(x), dim=1)
        shape_out = F.log_softmax(self.fc_shape(x), dim=1)
        loc_out = F.log_softmax(self.fc_loc(x), dim=1)
        return color_out, shape_out, loc_out


class Shape2DObjCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_out, obj_target):
        color_target = obj_target[:, 0]
        shape_target = obj_target[:, 1]
        loc_target = obj_target[:, 2]
        color_out, shape_out, loc_out = model_out
        loss = F.nll_loss(color_out, color_target) + \
               F.nll_loss(shape_out, shape_target) + \
               F.nll_loss(loc_out, loc_target)
        return loss


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def compute_accuracy(model_out, obj_target):
    obj_target = obj_target.data
    color_target = obj_target[:, 0]
    shape_target = obj_target[:, 1]
    loc_target = obj_target[:, 2]
    color_out, shape_out, loc_out = model_out[0].data, model_out[1].data, model_out[2].data,
    batch_size = obj_target.size(0)
    color_accuracy = torch.eq(torch.max(color_out, dim=1)[1], color_target).sum()/batch_size
    shape_accuracy = torch.eq(torch.max(shape_out, dim=1)[1], shape_target).sum()/batch_size
    loc_accuracy = torch.eq(torch.max(loc_out, dim=1)[1], loc_target).sum()/ batch_size
    return color_accuracy, shape_accuracy, loc_accuracy


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
    model.train()
    for batch_idx, (prev_canvas, inst, next_obj, final_canvas) in enumerate(train_loader):
        # for ix in range(args.batch_size):
        #     # ix = random.randint(0, args.batch_size-1)
        #     inst_str = ' '.join(map(train_loader.dataset.ix_to_word.get, list(inst[ix])))
        #     next_obj_color = train_loader.dataset.num2color[next_obj[ix][0]]
        #     next_obj_shape = train_loader.dataset.num2shape[next_obj[ix][1]]
        #     print(inst_str)
        #     print((next_obj_color, next_obj_shape))
        #     print("\n")
        prev_canvas = Variable(prev_canvas.cuda())
        inst = Variable(inst.cuda())
        next_obj = Variable(next_obj.cuda())
        output = model(inst, prev_canvas)
        loss = loss_fn(output, next_obj)
        loss.backward()
        clip_gradient(optimizer, 0.1)
        optimizer.step()
        color_accuracy, shape_accuracy, loc_accuracy = compute_accuracy(output, next_obj)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} ({:.3f} {:.3f} {})'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0], color_accuracy, shape_accuracy, loc_accuracy))


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
