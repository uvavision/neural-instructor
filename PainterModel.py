import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


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
        # TODO early stop when column contains all zero
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
        self.fc_act = nn.Linear(self.inst_encoder.rnn_size, 2)
        self.fc_color = nn.Linear(self.inst_encoder.rnn_size, 3)
        self.fc_shape = nn.Linear(self.inst_encoder.rnn_size, 3)
        self.fc_abs_loc = nn.Linear(self.inst_encoder.rnn_size, 2)
        self.fc_loc_route = nn.Sequential(nn.Linear(self.inst_encoder.rnn_size, 32), nn.ReLU(), nn.Linear(32, 2))
        self.fc_ref_obj = nn.Sequential(nn.Linear(68, 32), nn.ReLU(), nn.Linear(32, 1))
        self.fc_offset = nn.Linear(self.inst_encoder.rnn_size, 2)
        self.rewards = None
        self.saved_log_probs = None
        self.running_baseline = 0
        self.saved_actions = None

    def loc_relative_predict(self, inst_embedding, canvas):
        offset = F.hardtanh(self.fc_offset(inst_embedding))  # Bx2
        inst2 = torch.unsqueeze(inst_embedding, 1)  # Bx1x64
        inst2 = inst2.repeat(1, 25, 1) # Bx25x64
        att = torch.cat([inst2, canvas], 2) # Bx25x68
        att = self.fc_ref_obj(att).squeeze() # Bx25
        weight = F.softmax(att, dim=1)
        ref_obj = torch.bmm(weight.unsqueeze(1), canvas).squeeze(1)
        return ref_obj[:, 2:] + offset

    def forward(self, inst, prev_canvas, ref_obj, target_obj):
        inst_embedding = self.inst_encoder(inst) # Bx64
        act_out = F.log_softmax(self.fc_act(inst_embedding), dim=1)
        color_out = F.log_softmax(self.fc_color(inst_embedding), dim=1)
        shape_out = F.log_softmax(self.fc_shape(inst_embedding), dim=1)
        loc_abs = self.fc_abs_loc(inst_embedding)
        loc_relative = self.loc_relative_predict(inst_embedding, prev_canvas)
        weight = F.softmax(self.fc_loc_route(inst_embedding), dim=1)
        m = Categorical(weight)
        actions = m.sample()
        # print((torch.eq((ref_obj.sum(dim=1) < 0).long(), actions).sum().float() / ref_obj.size(0)).data[0])
        self.saved_log_probs = m.log_prob(actions)
        self.saved_actions = actions.data
        actions = actions.float()
        # # loc_out = loc_abs * weight[:, 0:1] + loc_relative * weight[:, 1:2]
        # actions = (ref_obj.sum(dim=1) < 0).float()
        loc_out = loc_abs * actions.unsqueeze(1) + loc_relative * (1 - actions).unsqueeze(1)
        return act_out, color_out, shape_out, loc_out[:, 0], loc_out[:, 1]


class Shape2DObjCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_out, act_target, target_obj, ref_obj):
        color_target = target_obj[:, 0]
        shape_target = target_obj[:, 1]
        act_out, color_out, shape_out, row_out, col_out = model_out
        loss = F.nll_loss(act_out, act_target) + \
               F.nll_loss(color_out, color_target) + \
               F.nll_loss(shape_out, shape_target) + \
               F.l1_loss(row_out, target_obj[:, 2].float()) + \
               F.l1_loss(col_out, target_obj[:, 3].float())
        # loss = F.nll_loss(color_out, color_target) + \
        #        F.nll_loss(shape_out, shape_target)
        rewards = F.l1_loss(row_out, target_obj[:, 2].float(), reduce=False).data + \
                  F.l1_loss(col_out, target_obj[:, 3].float(), reduce=False).data
        rewards = - rewards
        # model.rewards = rewards - model.running_baseline
        # model.running_baseline = 0.9 * model.running_baseline + 0.1 * rewards.mean()
        # model.rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        # model.rewards = (rewards - rewards.mean())
        rewards = (rewards - rewards.mean())
        return loss, rewards

def get_painter_model_prediction(painter_model, inst_samples, prev_canvas):
    assert inst_samples.size(1) == 21
    # [a, b, c, 0, d, 0] -> [a, b, c, 0, 0, 0]
    masks = (inst_samples == 0)
    for i in range(inst_samples.size(0)):
        if masks[i].sum() > 0:
            index = torch.nonzero(masks[i])[0, 0]
            inst_samples[i, index:] = 0
    samples_input = torch.zeros(inst_samples.size(0), inst_samples.size(1) + 2).long()
    # [a, b, ...] -> [0, a, b, ...]
    samples_input[:, 1:inst_samples.size(1) + 1] = inst_samples
    samples_input = Variable(samples_input.cuda())
    # vars = [Variable(var.data, volatile=True) for var in vars]
    output = painter_model(samples_input, prev_canvas, ref_obj=None, target_obj=None)
    act = torch.max(output[0].data, dim=1)[1]
    prediction = [p.data for p in output[1:]]
    target_obj = torch.stack([torch.max(prediction[0], dim=1)[1],
                              torch.max(prediction[1], dim=1)[1],
                              torch.round(prediction[2]).long(),
                              torch.round(prediction[3]).long()], dim=1)
    return act, target_obj

