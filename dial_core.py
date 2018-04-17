from const import *
import random
from collections import OrderedDict
from synthetic_layout import (Object, Canvas,
                              tmpl2txt_act, tmpl2txt_loc, tmpl2txt_obj)


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
        self.act = None
        self.objs = None
        self.obj = None
        self.loc_abs = None
        self.loc_rel = None
        self.obj_ref = None
        self.message = None
        self.mode_loc = None
        self.mode_style = None
        self.mode_ref = None
        self.is_viable = None

    def reset(self):
        self.objs = None
        self.obj = None
        self.loc_abs = None
        self.loc_rel = None
        self.obj_ref = None
        self.mode_ref = None
        self.message = None

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
        if act == ADD:
            self.get_add_activity()
            self.mode_ref = MODE_FULL
            self.message = self.generate_act_message_by_tmpl()
            self.canvas.add(self.obj)
        if act == DELETE:
            self.get_delete_activity()
            self.mode_ref = MODE_MIN
            self.message = self.generate_act_message_by_tmpl()
            self.canvas.delete(self.obj)
        if act == MOVE:
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
        if act == ADD:
            t_obj = self.canvas.get_obj_desc(self.obj, MODE_FULL)
        if act == DELETE:
            t_obj = self.canvas.get_obj_desc(self.obj, self.mode_ref)
        if act == MOVE:
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


if __name__ == '__main__':
    for i in range(100):
        agent = Agent()
        agent.mode_loc = LOC_REL
        agent.is_viable = True
        lst_activity = [ADD, ADD, DELETE, ADD, DELETE, DELETE]
        lst_activity = [ADD, MOVE]
        for act in lst_activity:
            agent.get_activity(act)
            # print(agent.act, agent.obj, agent.loc_abs, agent.loc_rel, agent.obj_ref)
            print("###", agent.message)
            agent.reset()
        # break
