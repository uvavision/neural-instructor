import sys

from const import *
import random
import json
from collections import OrderedDict, Counter
from layout import (Object, Canvas, tmpl2txt_act)
import requests
from tqdm import tqdm
from data_utils import *


class Policy(object):
    # TODO: load policy from domain config file
    def __init__(self, domain, path):
        self.domain = domain
        self.policy = self.load(path)

    def load(self, path=None):
        d = {}
        return d


class Agent(object):

    def __init__(self, mode_loc=None, mode_ref=None, is_viable=None, policy=None, domain=DOMAIN_2DSHAPE):
        self.domain = domain
        self.policy = policy
        self.states = OrderedDict()
        self.canvas = Canvas()
        # activity
        self.act = None  # the action - add, delete, move
        self.obj = None  # the object to be acted on
        self.loc_abs = None  # the name of the absolute location of obj
        self.loc_rel = None  # the name of the spatial relation between obj and another object
        self.obj_ref = None  # the referred object for the relative location
        self.message = None  # instruction or response or question
        # config
        self.mode_loc = mode_loc  # absolute or relative location
        self.mode_ref = mode_ref  # how to refer an object, full or min set of features
        self.is_viable = is_viable  # if the action is viable. set False to add noise

    def reset_activity(self):
        self.act = None
        self.obj = None
        self.loc_abs = None
        self.loc_rel = None
        self.obj_ref = None
        self.message = None

    def reset_config(self, mode_loc=None, mode_ref=None, is_viable=None):
        self.mode_loc = mode_loc
        self.mode_ref = mode_ref
        self.is_viable = is_viable

    def activity2dict(self):
        d = {'act': self.act,
             'obj': self.obj.to_dict() if self.obj else None,
             'loc_abs': self.loc_abs,
             'loc_rel': self.loc_rel,
             'obj_ref': self.obj_ref.to_dict() if self.obj_ref else None,
             'features': self.obj.features,
             'message': self.message
        }
        return d

    def config2dict(self):
        d = {'mode_loc': self.mode_loc, 'mode_ref': self.mode_ref, 'is_viable': self.is_viable}
        return d

    def get_add_activity(self, select_empty=True, is_viable=True, excluded_grids=[]):
        if self.canvas.size() == 0:
            self.mode_loc == LOC_ABS
        color = random.choice(COLORS)
        shape = random.choice(SHAPES)
        options = []
        loc_abs, row_abs, col_abs = self.canvas.select_loc_abs(select_empty, is_viable)
        if loc_abs:
            options.append(LOC_ABS)
        obj_ref, row_ref, col_ref, loc_rel = self.canvas.select_obj_ref_grid(select_empty, is_viable, excluded_grids)
        if obj_ref:
            options.append(LOC_REL)
        if len(options) == 0:
            return
        if self.mode_loc in options:
            mode_loc = self.mode_loc
        else:
            mode_loc = random.choice(options)
        if mode_loc == LOC_ABS:
            self.obj = Object(color, shape, row_abs, col_abs)
            self.loc_abs = loc_abs
        elif mode_loc == LOC_REL:
            self.obj = Object(color, shape, row_ref, col_ref)
            self.obj_ref = obj_ref
            self.loc_rel = loc_rel
        '''
        elif self.mode_style == PATTERN:
            style = random.choice(PATTERN_STYLE)
            if style == 'row':
                obj1 = Object(color, shape, row, col - 1)
                obj2 = Object(color, shape, row, col)
                obj3 = Object(color, shape, row, col + 1)
            if style == 'column':
                obj1 = Object(color, shape, row - 1, col)
                obj2 = Object(color, shape, row, col)
                obj3 = Object(color, shape, row + 1, col)
            self.objs = [obj1, obj2, obj3]
        '''

    def get_delete_activity(self, select_empty=False, is_viable=True):
        assert self.canvas.size() > 0
        if self.canvas.size() == 1:
            self.obj = list(self.canvas.d_id_obj.values())[0]
            return
        if not self.mode_loc:
            self.mode_loc = random.choice(MODES_LOC)
        self.loc_abs, row, col = self.canvas.select_loc_abs(select_empty, self.is_viable)
        if not row:
            self.obj_ref, row, col, self.loc_rel = self.canvas.select_obj_ref_grid(select_empty, is_viable)
        if row and col:
            for id_, obj in self.canvas.d_id_obj.items():
                if (row, col) == (obj.row, obj.col):
                    self.obj = obj
                    break
        if not self.obj:
            self.obj = random.choice(list(self.canvas.d_id_obj.values()))

    # def get_move_activity(self, select_empty_del=False, select_empty_add=True):
    #     assert self.canvas.size() > 0
    #     self.get_delete_activity(select_empty=select_empty_del)
    #     obj_from = self.obj
    #     self.get_add_activity(select_empty=select_empty_add, is_viable=self.is_viable, excluded_grids=[(obj_from.row, obj_from.col)])
    #     obj_to = Object(obj_from.color, obj_from.shape, self.obj.row, self.obj.col)
    #     return obj_from, obj_to

    def get_activity(self, act):
        if not self.mode_loc:
            self.mode_loc = random.choice(MODES_LOC)
        activities = []
        if act == ADD:
            self.act = act
            self.get_add_activity(select_empty=True, is_viable=self.is_viable)
            self.message = self.generate_act_message_by_tmpl()
            self.canvas.add(self.obj)
            activities.append(self.activity2dict())
        if act == DELETE:
            self.act = act
            self.get_delete_activity(select_empty=False, is_viable=self.is_viable)
            self.message = self.generate_act_message_by_tmpl()
            self.canvas.delete(self.obj)
            activities.append(self.activity2dict())
        if act == MOVE:
            self.get_delete_activity(select_empty=False, is_viable=self.is_viable)
            self.act = DELETE
            activities.append(self.activity2dict())
            obj_from = self.obj
            self.get_add_activity(select_empty=True, is_viable=self.is_viable,
                                  excluded_grids=[(obj_from.row, obj_from.col)])
            self.act = ADD
            self.obj.color, self.obj.shape = obj_from.color, obj_from.shape
            activities.append(self.activity2dict())
            obj_to = Object(self.obj.color, self.obj.shape, self.obj.row, self.obj.col)
            self.obj = obj_from
            self.message = self.generate_act_message_by_tmpl(MOVE)
            self.canvas.delete(obj_from)
            self.canvas.add(obj_to)
        self.canvas.update_ref_obj_features()
        return activities

    def generate_act_message_by_tmpl(self, act=None):
        if not act:
            act = self.act
        lst = []
        t_loc_abs = t_loc_rel = ''
        if act == ADD:
            t_obj = self.canvas.get_obj_desc(self.obj, MODE_FULL, self.mode_ref)
        if act == DELETE:
            t_obj = self.canvas.get_obj_desc(self.obj, self.mode_ref, self.mode_ref)
        if act == MOVE:
            t_obj = self.canvas.get_obj_desc(self.obj, self.mode_ref, self.mode_ref)
            if self.loc_abs:
                t_loc_abs = self.canvas.get_loc_desc(self.loc_abs, self.loc_rel, self.obj_ref, self.mode_ref)
            if self.loc_rel and self.obj_ref:
                t_loc_rel = self.canvas.get_loc_desc(self.loc_abs, self.loc_rel, self.obj_ref, self.mode_ref)
        message = tmpl2txt_act(act, t_obj, t_loc_abs, t_loc_rel)
        return message

    def get_next_message(self, actions_pre, dial_act_pre, role_pre):
        actions = []
        role = OTHER_ROLE[role_pre]
        return role, actions_pre


def get_act_sequence(n_obj):
    lst_act = [ADD]
    n_delete = random.randint(0, 3)
    n_add = n_obj + n_delete
    n_move = random.randint(0, 2)
    n_turn = n_move + n_delete + n_add
    # print('#obj: {}, #turn: {}, #add: {}, #delete: {}, #move: {}'.format(n_obj, n_turn, n_add, n_delete, n_move))
    while len(lst_act) < n_turn:
        d_ct = Counter(lst_act)
        ct_add, ct_delete, ct_move = d_ct[ADD], d_ct[DELETE], d_ct[MOVE]
        if ct_move < n_move and ct_add > ct_delete and random.choice([True, False]):
            lst_act.append(MOVE)
        if ct_delete < n_delete and ct_delete < ct_add and random.choice([True, False]):
            lst_act.append(DELETE)
        if ct_add < n_add and random.choice([True, False]):
            lst_act.append(ADD)
    return lst_act


def is_level3_ref(instruction):
    vals_abs = DICT_LOC_ABS2NAME.values()
    vals_rel = DICT_LOC_DELTA2NAME.values()
    count = 0
    for w in instruction.split():
        if w in vals_abs or w in vals_rel:
            count += 1
    assert count <= 3
    return count == 3


def generate_data(n_dial, is_viable=True, mode_ref=MODE_MIN, out_json=None):
    data = []
    count = 0
    for i in tqdm(range(n_dial)):
        d_dial = {'dial_id': i + 1, 'dialog_data': []}
        n_obj = random.randint(3, 6)
        lst_act = get_act_sequence(n_obj)
        # FIXME
        # lst_act = [ADD] * 10
        while 'move' in lst_act:
            lst_act.remove('move')
        assert not 'move' in lst_act
        while 'delete' in lst_act:
            lst_act.remove('delete')
        assert not 'delete' in lst_act
        agent = Agent(mode_loc=None, mode_ref=mode_ref, is_viable=is_viable)
        visualize_samples = []
        # turn3_rel_ref = False
        # turn2_rel_ref = False
        ref_types = []
        target_ref_types = ['abs', 'abs', 'rel', 'abs']
        max_turns = len(target_ref_types)
        lst_act = lst_act[:max_turns]
        for turn, act in enumerate(lst_act):
            canvas_data = [v.get_info() for k, v in agent.canvas.d_id_obj.items()]
            activities = agent.get_activity(act)
            current_canvas = [v.get_info() for k, v in agent.canvas.d_id_obj.items()]
            # if is_level3_ref(activities[0]['message']) or random.random() < 0.2:
            d = {'turn': turn + 1, 'config': agent.config2dict(), 'activities': activities,
                 # 'prev_canvas': canvas_data,
                 'current_canvas': current_canvas}
            d_dial['dialog_data'].append(d)
            count += 1
            # print('@@@', agent.act, agent.obj, agent.loc_abs, agent.loc_rel, agent.obj_ref)
            # print("###", agent.message)
            # print(agent.canvas.get_desc())
            visualize_samples.append({"instruction": agent.message, "canvas": agent.canvas.get_desc()})
            agent.reset_activity()
            agent.reset_config(mode_loc=None, mode_ref=mode_ref, is_viable=is_viable)
            ref_type = inst_ref_type(activities[0]['message'])
            if ref_type != INST_REL:
                activities[0]['features']['obj_ref'] = None
            if ref_type == INST_REL:
                ref_types.append("rel")
            else:
                ref_types.append("abs")
            # if turn + 1 == 2 and inst_ref_type(activities[0]['message']) == INST_REL:
            #     turn2_rel_ref = True
            #     assert activities[0]['features']['obj_ref'] is not None
            # # if turn + 1 == 2 and not turn2_rel_ref:
            # #     print(activities[0]['message'])
            # #     assert activities[0]['features']['obj_ref'] is None
            # if turn + 1 == 3 and inst_ref_type(activities[0]['message']) == INST_REL:
            #     turn3_rel_ref = True
            #     break
            if turn + 1 == max_turns:
                break
        # if ref_types == ['abs', 'abs', 'rel', 'rel', 'abs']:
        if ref_types == target_ref_types:
            data.append(d_dial)
        # run the server: python canvas_render.py, visit the result at http://localhost:5001/dialog
        # r = requests.post("http://vision.cs.virginia.edu:5001/new_dialog", json=visualize_samples)
        # r.raise_for_status()
    print("numer of instructions: {}".format(count))
    if out_json:
        print("number of dialogs: {}".format(len(data)))
        json.dump(data, open(out_json, 'w'), indent=2)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        generate_data(500000, out_json=sys.argv[1])
    else:
        generate_data(100000)

