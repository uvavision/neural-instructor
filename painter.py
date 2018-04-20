from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from Shape2D import get_shape2d_loader
from PainterModel import Shape2DPainterNet, Shape2DObjCriterion
import random
from utils import clip_gradient, adjust_lr

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
# model.load_state_dict(torch.load('painter-models_simple_mean/model_20.pth'))
model.cuda()
loss_fn = Shape2DObjCriterion()

# model_fc_loc_route = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
# model_fc_loc_route.cuda()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
# optimizer = optim.Adam(model_fc_loc_route.parameters(), lr=args.lr, weight_decay=0)


def train(epoch):
    # adjust_lr(optimizer, epoch, args.lr)
    model.train()
    for batch_idx, (prev_canvas, inst, next_obj, final_canvas, ref_obj, raw_data) in enumerate(train_loader):
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
        optimizer.zero_grad()
        output = model(inst, prev_canvas, ref_obj, next_obj)
        loss, rewards = loss_fn(output, next_obj, ref_obj)
        policy_loss = (-model.saved_log_probs * Variable(rewards)).sum()
        (loss + policy_loss).backward()
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
    torch.save(model.state_dict(), 'painter-models_simple_mean_validate/model_{}.pth'.format(epoch))
    # torch.save(optimizer.state_dict(), 'painter-models/optimizer_{}.pth'.format(epoch))
    # model_test()
