from const import *
import random
import json
from collections import OrderedDict, Counter
from layout import (Object, Canvas, tmpl2txt_act)


class Policy(object):
    # TODO: load policy from domain config file
    def __init__(self, domain, path):
        self.domain = domain
        self.policy = self.load(path)

    def load(self, path=None):
        d = {}
        return d


class Agent(object):

    def __init__(self, policy=None, domain=DOMAIN_2DSHAPE):
        self.domain = domain
        self.policy = policy
        self.states = OrderedDict()
        self.canvas = Canvas()
        # activity
        self.act = None  # the action - add, delete,move
        self.obj = None  # the object to be acted on
        self.loc_abs = None  # the name of the absolute location of obj
        self.loc_rel = None  # the name of the spatial relation between obj and another object
        self.obj_ref = None  # the referred object for the relative location
        self.message = None  # instruction or response or questions
        # config
        self.mode_loc = None  # absolute or relative location
        self.mode_ref = None  #
        self.mode_style = None  #
        self.is_viable = None  # if the action is viable. set False to add noise

    def reset_activity(self):
        self.act = None
        self.obj = None
        self.loc_abs = None
        self.loc_rel = None
        self.obj_ref = None
        self.features = None
        self.message = None

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
        d = {}
        return d

    def reset_config(self):
        self.mode_loc = None
        self.mode_ref = None
        self.mode_style = None
        self.is_viable = None

    def get_add_activity(self, select_empty=True, is_viable=True, exclude_grids=[]):
        if self.canvas.size() == 0:
            self.mode_loc == LOC_ABS
        color = random.choice(COLORS)
        shape = random.choice(SHAPES)
        loc_abs, obj_ref, loc_rel = None, None, None
        options = []
        loc_abs, row_abs, col_abs = self.canvas.select_loc_abs(select_empty, is_viable)
        if loc_abs:
            options.append(LOC_ABS)
        obj_ref, row_ref, col_ref, loc_rel = self.canvas.select_loc_rel(select_empty, is_viable, exclude_grids)
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

    def get_delete_activity(self, select_empty=False):
        assert self.canvas.size() > 0
        if self.canvas.size() == 1:
            self.obj = list(self.canvas.d_id_obj.values())[0]
            return
        if not self.mode_loc:
            self.mode_loc = random.choice(MODES_LOC)
        row = col = None
        self.loc_abs, row, col = self.canvas.select_loc_abs(select_empty, self.is_viable)
        if not row:
            self.obj_ref, self.loc_rel, row, col = self.canvas.select_loc_rel(select_empty, self.is_viable)
        if row and col:
            for id_, obj in self.canvas.d_id_obj.items():
                if (row, col) == (obj.row, obj.col):
                    self.obj = obj
                    break
        if not self.obj:
            self.obj = random.choice(list(self.canvas.d_id_obj.values()))

    def get_move_activity(self):
        assert self.canvas.size() > 0
        obj_from, obj_to = None, None
        self.get_delete_activity()
        obj_from = self.obj
        self.get_add_activity(select_empty=True, is_viable=True, exclude_grids=[(obj_from.row, obj_from.col)])
        obj_to = Object(obj_from.color, obj_from.shape, self.obj.row, self.obj.col)
        return obj_from, obj_to

    def get_activity(self, act):
        self.act = act
        if not self.mode_style:
            self.mode_style = random.choice(MODES_STYLE)
        if not self.mode_loc:
            self.mode_loc = random.choice(MODES_LOC)
        self.mode_style = SINGLE
        self.mode_loc = LOC_ABS
        if self.act == ADD:
            self.get_add_activity()
            self.mode_ref = MODE_FULL
            self.message = self.generate_act_message_by_tmpl()
            self.canvas.add(self.obj)
        if self.act == DELETE:
            self.get_delete_activity()
            self.mode_ref = MODE_MIN
            self.message = self.generate_act_message_by_tmpl()
            self.canvas.delete(self.obj)
        if self.act == MOVE:
            obj_from, obj_to = self.get_move_activity()
            self.obj = obj_from
            self.message = self.generate_act_message_by_tmpl()
            self.canvas.delete(obj_from)
            self.canvas.add(obj_to)

    def generate_act_message_by_tmpl(self):
        if not self.mode_ref:
            self.mode_ref = random.choice(MODES_REF)
        lst = []
        t_loc_abs = t_loc_rel = ''
        if self.act == ADD:
            t_obj = self.canvas.get_obj_desc(self.obj, MODE_FULL)
        if self.act == DELETE:
            t_obj = self.canvas.get_obj_desc(self.obj, self.mode_ref)
        if self.act == MOVE:
            t_obj = self.canvas.get_obj_desc(self.obj, self.mode_ref)
            if self.loc_abs:
                t_loc_abs = self.canvas.get_loc_desc(self.loc_abs, self.loc_rel, self.obj_ref)
            if self.loc_rel and self.obj_ref:
                t_loc_rel = self.canvas.get_loc_desc(self.loc_abs, self.loc_rel, self.obj_ref)
        message = tmpl2txt_act(self.act, t_obj, t_loc_abs, t_loc_rel)
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
    print('#obj: {}, #turn: {}, #add: {}, #delete: {}, #move: {}'.format(n_obj, n_turn, n_add, n_delete, n_move))
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


def generate_data(n_dial, out_json=None):
    data = []
    for i in range(n_dial):
        d_dial = {'dial_id': i + 1, 'dialog_data': []}
        n_obj = random.randint(3, 6)
        lst_act = get_act_sequence(n_obj)
        print(lst_act)
        agent = Agent()
        for turn, act in enumerate(lst_act):
            agent.get_activity(act)
            d = {'turn': turn + 1,'config': agent.config2dict(), 'activity': agent.activity2dict()}
            d_dial['dialog_data'].append(d)
            # print(agent.act, agent.obj, agent.loc_abs, agent.loc_rel, agent.obj_ref)
            print("###", agent.message)
            agent.reset_activity()
            agent.reset_config()
        data.append(d_dial)
    if out_json:
        json.dump(data, open(out_json, 'w'), indent=4)


if __name__ == '__main__':
    generate_data(10, 'data_dial.json')
