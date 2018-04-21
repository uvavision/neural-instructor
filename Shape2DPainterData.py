from __future__ import print_function

import datetime
import torch.utils.data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
from pprint import pprint
from pymongo import MongoClient
from os import sys, path
from tqdm import tqdm


# %%
def render_canvas(objects):
    grid_size = 100
    layout = []
    for obj in objects:
        top = obj['row'] * grid_size + 10
        left = obj['col'] * grid_size + 10
        width = grid_size - 20
        height = grid_size - 20
        label = obj['color']
        shape = obj['shape']
        if shape == 'square':
            shape = 'rectangle'
        layout.append({"left": left, "top": top, "width": width, "height": height, "label": label, "shape": shape})
    return '#CANVAS-' + str(layout).replace("'", '"').replace(' ', '')


# %%
class Shape2DPainterData(torch.utils.data.Dataset):
    def __init__(self, datafile):
        data = json.load(open(datafile, 'r'))
        # self.data = [d for d in data if 'canvas' in d['current_instruction']]
        self.data = []
        max_seq_length = 0
        vocab = set()
        for dialog in data:
            for turn in dialog['dialog_data']:
                assert len(turn['activities']) == 1
                activity = turn['activities'][0]
                assert activity['act'] in ['add', 'delete']
                message = activity['message']
                words = message.split()
                vocab = vocab.union(set(words))
                max_seq_length = max(max_seq_length, len(words))
                self.data.append({'instruction': message,
                                  'target_obj': activity['obj'],
                                  'prev_canvas': turn['prev_canvas'],
                                  'act': activity['act']})
        self.vocab = sorted(list(vocab))
        self.vocab_size = len(self.vocab)
        self.max_seq_length = max_seq_length
        self.word_to_ix = {self.vocab[i]: i+1 for i in range(len(self.vocab))}
        self.ix_to_word = {v: k for k, v in self.word_to_ix.items()}
        self.ix_to_word[0] = 'END'
        self.color2num = {'red': 0, 'green': 1, 'blue': 2}
        self.shape2num = {'square': 0, 'circle': 1, 'triangle': 2}
        self.num2color = {v: k for k, v in self.color2num.items()}
        self.num2shape = {v: k for k, v in self.shape2num.items()}

    def __getitem__(self, index):
        raw_data = self.data[index]
        prev_canvas_data = self.encode_canvas(raw_data['prev_canvas'])
        inst_data = self.encode_inst(raw_data['instruction'])
        target_obj_data = self.encode_obj(raw_data['target_obj']).astype(np.int64)
        act_data = self.encode_act(raw_data['act'])
        return prev_canvas_data, inst_data, target_obj_data, act_data, raw_data
        # final_canvas_data = self.encode_canvas(raw_data['final_canvas'])
        # if 'ref_obj' in raw_data:
        #     ref_obj = self.encode_obj(raw_data['ref_obj']).astype(np.int64)
        #     a = prev_canvas_data[ref_obj[-2]*5+ref_obj[-1]].astype(np.int64)
        #     assert np.array_equal(a, ref_obj)
        # else:
        #     ref_obj = np.array([-1, -1, -1, -1], dtype=np.int64)
        # return prev_canvas_data, inst_data, target_obj_data, final_canvas_data, ref_obj, raw_data

    def get_raw_item(self, index):
        return self.data[index]

    def encode_act(self, act):
        return {'add': 0, 'delete': 1}[act]

    def encode_inst(self, inst_str):
        # [0, a, b, c, 0]
        inst_data = np.zeros(self.max_seq_length + 2, dtype=np.int64)
        inst_indices = [self.word_to_ix[w] for w in inst_str.split()]
        inst_data[1:len(inst_indices) + 1] = inst_indices
        return inst_data

    def encode_obj(self, obj):
        return np.array((self.color2num[obj['color']],
                         self.shape2num[obj['shape']],
                         obj['row'], obj['col']), dtype=np.int32)

    def decode_obj_arr(self, obj_arr):
        color, shape, row, col = obj_arr.long()
        return {'color': self.num2color[color], 'shape': self.num2shape[shape],
                'row': row, 'col': col}

    def decode_canvas(self, canvas):
        objs = []
        for i in range(25):
            if canvas[i].sum() > 0:
                objs.append(self.decode_obj_arr(canvas[i]))
        return objs

    def encode_canvas(self, canvas_objs):
        canvas_data = np.zeros((25, 4), np.float32)
        canvas_data.fill(-1)
        for obj in canvas_objs:
            obj_encode = self.encode_obj(obj)
            canvas_data[obj_encode[-2]*5+obj_encode[-1]] = obj_encode
        return canvas_data

    def __len__(self):
        return len(self.data)


def shape2d_painter_data_collate(data):
    prev_canvas_data, inst_data, target_obj_data, act_data, raw_data = zip(*data)
    return torch.from_numpy(np.stack(prev_canvas_data)), \
           torch.from_numpy(np.stack(inst_data)), \
           torch.from_numpy(np.stack(target_obj_data)), \
           torch.from_numpy(np.array(act_data, dtype=np.int64)), \
           raw_data


def get_shape2d_painter_data_loader(split, batch_size):
    assert split in ['train', 'val', 'sample']
    datafile = {'train': 'train.json', 'val': 'val.json', 'sample': 'sample.json'}[split]
    dataset = Shape2DPainterData(datafile)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=shape2d_painter_data_collate)
    return dataloader


def dataset_test():
    #%%
    dataset = Shape2DPainterData('sample.json')
    for i in range(len(dataset)):
        prev_canvas_data, inst_data, target_obj_data, act_data, raw_data = dataset[i]
        print("previous canvas:")
        pprint(prev_canvas_data)
        print("instruction data:")
        pprint(inst_data)
        print("target obj data:")
        pprint(target_obj_data)
        print("raw data:")
        pprint(raw_data)
        pprint("act data:")
        pprint(act_data)
        print('====================')


def dataloader_test():
    batch_size = 3
    dataloader = get_shape2d_painter_data_loader(split='sample', batch_size=batch_size)
    prev_canvas_data, inst_data, target_obj_data, act_data, raw_data = next(iter(dataloader))
    assert prev_canvas_data.size(0) == batch_size


if __name__ == '__main__':
    # dataset_test()
    dataloader_test()
    # render_test()

