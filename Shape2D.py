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
class Shape2D(torch.utils.data.Dataset):
    def __init__(self, datafile):
        data = json.load(open(datafile, 'r'))
        # self.data = [d for d in data if 'canvas' in d['current_instruction']]
        self.data = data
        max_seq_length = 0
        vocab = set()
        for d in self.data:
            words = d['current_instruction'].split()
            vocab = vocab.union(set(words))
            max_seq_length = max(max_seq_length, len(words))
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
        inst_data = self.encode_inst(raw_data['current_instruction'])
        next_obj_data = self.encode_obj(raw_data['next_object']).astype(np.int64)
        final_canvas_data = self.encode_canvas(raw_data['final_canvas'])
        # ref_obj = np.array((0, 0, 0, 0), dtype=np.int64)
        # assert 'ref_obj' in raw_data
        # if 'ref_obj' in raw_data:
        ref_obj = self.encode_obj(raw_data['ref_obj']).astype(np.int64)
        a = prev_canvas_data[ref_obj[-2]*5+ref_obj[-1]].astype(np.int64)
        assert  np.array_equal(a, ref_obj)
        return prev_canvas_data, inst_data, next_obj_data, final_canvas_data, ref_obj

    def get_raw_item(self, index):
        return self.data[index]

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

    def encode_canvas(self, canvas_objs):
        canvas_data = np.zeros((25, 4), np.float32)
        canvas_data.fill(-1)
        for obj in canvas_objs:
            obj_encode = self.encode_obj(obj)
            canvas_data[obj_encode[-2]*5+obj_encode[-1]] = obj_encode
        return canvas_data

    def __len__(self):
        return len(self.data)


def shape2d_collate(data):
    prev_canvas_data, inst_data, next_obj_data, final_canvas_data, ref_obj_data = zip(*data)
    return torch.from_numpy(np.stack(prev_canvas_data)), \
           torch.from_numpy(np.stack(inst_data)), \
           torch.from_numpy(np.stack(next_obj_data)), \
           torch.from_numpy(np.stack(final_canvas_data)), \
           torch.from_numpy(np.stack(ref_obj_data))


def get_shape2d_loader(split, batch_size):
    assert split in ['train', 'val', 'sample']
    datafile = {'train': 'train.json', 'val': 'val.json', 'sample': 'sample.json'}[split]
    dataset = Shape2D(datafile)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=shape2d_collate)
    return dataloader


def dataset_test():
    #%%
    dataset = Shape2D('sample.json')
    pprint(dataset.vocab)
    d = dataset.get_raw_item(0)
    pprint(d)
    prev_canvas_data, inst_data, next_obj_data, final_canvas_data = dataset[0]
    pprint(prev_canvas_data)
    pprint(inst_data)
    pprint(next_obj_data)
    pprint(final_canvas_data)


def dataloader_test():
    dataloader = get_shape2d_loader(split='sample', batch_size=3)
    prev_canvas_data, inst_data, next_obj_data, final_canvas_data = next(iter(dataloader))
    print('aka')


def render_test():
    config = None
    config_file = '/net/zf14/xy4cm/Projects/chat-crowd/main/domains/app-2Dshape.json'
    with open(config_file) as f:
        config = json.load(f)
    uri_remote = (config['compose-for-mongodb'][0]['credentials']['uri'] +
                  '&ssl_cert_reqs=CERT_NONE')
    cli = MongoClient(uri_remote)
    mockdb = config["domain-db"]["db-name"]
    coll_chat = cli[mockdb][config["domain-db"]["coll_chat_data"]]
    print(coll_chat)
    task_id = '12557'
    coll_chat.delete_many({'task_id': task_id})
    dataset = Shape2D('sample.json')
    for index in tqdm(range(len(dataset))):
        data = dataset.get_raw_item(index)
        inst = data['current_instruction']
        coll_chat.insert_one({'msg': inst, 'author': 'human', 'task_id': task_id,
                          'username': 'aaa', 'worker_id': '123456', "role": 'user', 'turn': str(index), 'mode': '2Dshape',
                          'timestamp': str(datetime.datetime.now())})
        objs = data['prev_canvas']
        objs.append(data['next_object'])
        coll_chat.insert_one({'msg': render_canvas(objs), 'author': 'human', 'task_id': task_id,
                          'username': 'bbb', 'worker_id': '123456', "role": 'agent', 'turn': str(index), 'mode': '2Dshape',
                          'timestamp': str(datetime.datetime.now())})


if __name__ == '__main__':
    # dataloader_test()
    render_test()

