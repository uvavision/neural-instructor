from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from Shape2DPainterOmniData import get_shape2d_painter_data_loader
from PainterModelOmni import Shape2DPainterNet, Shape2DObjCriterion
import random
from utils import clip_gradient, adjust_lr

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def compute_accuracy(model_out, act_target, target_obj, ref_obj):
    color_target = target_obj[:, 0]
    shape_target = target_obj[:, 1]
    act_out, color_out, shape_out, row_out, col_out = model_out
    # ref_obj_target = ref_obj[:, -2] * 5 + ref_obj[:, -1]
    batch_size = target_obj.size(0)
    act_accuracy = torch.eq(torch.max(act_out.data, dim=1)[1], act_target.data).sum()/batch_size
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
    return act_accuracy, color_accuracy, shape_accuracy, row_accuracy.data[0], col_accuracy.data[0]


vocab = ['a', 'add', 'at', 'blue', 'bottom-left', 'bottom-left-of', 'bottom-middle', 'bottom-of', 'bottom-right',
         'bottom-right-of', 'canvas', 'center', 'circle', 'green', 'left-middle', 'left-of', 'location', 'now',
         'object', 'of', 'one', 'place', 'red', 'right-middle', 'right-of', 'square', 'the', 'to/at', 'top-left',
         'top-left-of', 'top-middle', 'top-of', 'top-right', 'top-right-of', 'triangle']
train_loader = get_shape2d_painter_data_loader(split='train', batch_size=args.batch_size)
test_loader = get_shape2d_painter_data_loader(split='val', batch_size=1)
print("vocab-size: {}".format(train_loader.dataset.vocab_size))
print(train_loader.dataset.vocab)
assert train_loader.dataset.vocab_size == test_loader.dataset.vocab_size
assert train_loader.dataset.max_seq_length == test_loader.dataset.max_seq_length
# assert train_loader.dataset.vocab == vocab
# assert test_loader.dataset.vocab == vocab

model = Shape2DPainterNet(train_loader.dataset.vocab_size)
# model.load_state_dict(torch.load('painter_model_new/model_20.pth'))
# model.load_state_dict(torch.load('painter_model_new_3way/model_20.pth'))
# model.load_state_dict(torch.load('painter_model_new_level2_continue/model_20.pth'))
# model.load_state_dict(torch.load('painter_model_new_level3/model_20.pth'))
# model.load_state_dict(torch.load('painter_model_new_level_combined//model_20.pth'))
# model.load_state_dict(torch.load('painter_model_new_add_remove///model_20.pth'))
# model.load_state_dict(torch.load('painter-omni///model_200.pth'))
# model.load_state_dict(torch.load('painter-omni-combine///model_19.pth'))
# model.load_state_dict(torch.load('painter-omni-combine-exp///model_20.pth'))
# model.load_state_dict(torch.load('painter-omni-abs_rel/model_2.pth'))
# model.load_state_dict(torch.load('painter-abs_abs_rel_abs/model_200.pth'))
# model.load_state_dict(torch.load('painter-omni-abs_abs_rel2/model_33.bak.pth'))
# model.load_state_dict(torch.load('painter-omni-abs_abs2_rel-separate//model_17.pth'))
# model.load_state_dict(torch.load('painter-omni-abs_abs2_rel_retrain/model_17.pth'))
# model.load_state_dict(torch.load('painter-omni-abs_abs2_rel_retrain_success_reward//model_16.pth'))
# model.load_state_dict(torch.load('step_gt_canvas//model_10.pth'))
# model.load_state_dict(torch.load('boostrap//model_118.pth'))
model.cuda()
# loss_fn = Shape2DObjCriterion()

# model_fc_loc_route = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

# model_fc_loc_route.cuda()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
# optimizer = optim.Adam(model_fc_loc_route.parameters(), lr=args.lr, weight_decay=0)

def adjust_lr(optimizer, epoch, initial_lr, decay_rate=0.8):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.8 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch):
    # adjust_lr(optimizer, epoch, args.lr)
    model.train()
    for batch_idx, dialog in enumerate(train_loader):
        optimizer.zero_grad()
        success = model(dialog, train_loader.dataset.ix_to_word)
        losses = []
        use_success_reward = False
        if use_success_reward:
            for log_prob, reward in zip(model.color_log_probs, model.color_rewards):
                losses.append((-log_prob * Variable(model.success_reward.cuda())).sum())
            for log_prob, reward in zip(model.shape_log_probs, model.shape_rewards):
                losses.append((-log_prob * Variable(model.success_reward.cuda())).sum())
            for log_prob, reward in zip(model.loc_log_probs, model.loc_rewards):
                losses.append((-log_prob * Variable(model.success_reward.cuda())).sum())
            for log_prob, reward in zip(model.att_log_probs, model.loc_rewards):
                if log_prob is not None:
                    losses.append((-log_prob * Variable(model.success_reward.cuda())).sum())
            sum(losses).backward()
        else:
            for log_prob, reward in zip(model.color_log_probs, model.color_rewards):
                losses.append((-log_prob * Variable(reward.cuda())).sum())
            for log_prob, reward in zip(model.shape_log_probs, model.shape_rewards):
                losses.append((-log_prob * Variable(reward.cuda())).sum())
            for log_prob, reward in zip(model.loc_log_probs, model.loc_rewards):
                losses.append((-log_prob * Variable(reward.cuda())).sum())
            for log_prob, reward in zip(model.att_log_probs, model.loc_rewards):
                if log_prob is not None:
                    losses.append((-log_prob * Variable(reward.cuda())).sum())
            sum(losses).backward()

        clip_gradient(optimizer, 0.1)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            total_steps = len(model.loc_rewards)
            fmt_template = "c{0}:{1:>6.3f}, s{0}:{2:>6.3f}, l{0}:{3:>6.3f} | "
            reward_report = ''
            for i in range(total_steps):
                reward_report += fmt_template.format(i, model.color_rewards[i].mean(),
                                                     model.shape_rewards[i].mean(), model.loc_rewards[i].mean())
            for i in range(len(model.att_rewards)):
                if model.att_rewards[i] is not None:
                    reward_report += "att:{:6.3f}|".format(model.att_rewards[i].mean())
            reward_report += " success: {:.3f}".format(success)
            print('E:{:3} [{:>6}/{} ({:>2.0f}%)]{}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), reward_report))

def model_test():
    model.eval()
    results = []
    for batch_idx, dialog in enumerate(test_loader):
        success = model(dialog, test_loader.dataset.ix_to_word)
        results.append(success)
        reward_report = " success: {:.3f}".format(success)
        print('[{:>6}/{} ({:>2.0f}%)]{} {:.3f}'.format(batch_idx, len(test_loader.dataset),
                                                100. * batch_idx / len(test_loader),
                                                       reward_report, sum(results)/(batch_idx+1)))


model_test()

# for epoch in range(1, args.epochs + 1):
#     train(epoch)
#     # torch.save(model.state_dict(), 'boostrap_dont_consider_rel_reward_when_prev_canvas_is_wrong/model_{}.pth'.format(epoch))
#     # torch.save(model.state_dict(), 'step_gt_canvas/model_{}.pth'.format(epoch))
#     # torch.save(model.state_dict(), 'dont_consider_rel_neg_reward_when_prev_canvas_is_wrong/model_{}.pth'.format(epoch))
#     # torch.save(model.state_dict(), 'painter-omni-abs_abs2_rel_success_reward_bootstrap/model_{}.pth'.format(epoch))
#     # torch.save(model.state_dict(), 'painter-omni-abs_abs2_rel_retrain_success_reward/model_{}.pth'.format(epoch))
#     # torch.save(model.state_dict(), 'painter-omni-abs_abs2_rel_retrain/model_{}.pth'.format(epoch))
#     # torch.save(model.state_dict(), 'painter-omni-abs_abs2_rel-separate/model_{}.pth'.format(epoch))
#     # torch.save(model.state_dict(), 'painter-omni-abs_abs_rel_resampled-shared_encoder/model_{}.pth'.format(epoch))
#     # torch.save(model.state_dict(), 'painter-omni-abs_abs_rel2/model_{}.pth'.format(epoch))
#     # torch.save(model.state_dict(), 'painter-omni-combine-exp-all/model_{}.pth'.format(epoch))
#     # torch.save(model.state_dict(), 'painter-omni-abs_rel/model_{}.pth'.format(epoch))
#     # torch.save(model.state_dict(), 'painter-omni-continue/model_{}.pth'.format(epoch))
# # #     # torch.save(optimizer.state_dict(), 'painter-models/optimizer_{}.pth'.format(epoch))
#
