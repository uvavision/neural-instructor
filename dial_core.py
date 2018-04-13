from const import *
import random
from collections import OrderedDict
from synthetic_layout import (Object, Canvas, get_add_activity,
                              get_delete_activity, get_move_activity,
                              tmpl2txt_act, tmpl2txt_loc, tmpl2txt_obj)


class Policy(object):
    def __init__(self, domain, path):
        self.domain = domain
        self.policy = load(path)

    def load(self, path=None):
        d = {}
        return d


class Agent(object):

    def __init__(self, policy=None, domain=DOMAIN_2DSHAPE):
        # TODO: load policy from domain config file
        self.domain = domain
        self.policy = policy
        self.states = OrderedDict()
        self.canvas = Canvas()

    def get_activity(self, act=ADD, mode_loc=None, mode_style=None):
        if not mode_style:
            mode_style = random.choice(MODES_STYLE)
        if not mode_loc:
            mode_loc = random.choice(MODES_LOC)
        mode_style = SINGLE
        mode_loc = LOC_ABS
        print(mode_style, mode_loc)
        if act == ADD:
            activity = get_add_activity(self.canvas, mode_loc, mode_style)
            self.canvas.add(activity[0])
        if act == DELETE:
            activity = get_delete_activity(self.canvas, mode_loc, mode_style)
            self.canvas.delete(activity[0])
        if act == MOVE:
            activity = get_move_activity(self.canvas, mode_loc, mode_style)
        return (act,) + activity

    def generate_act_message_by_tmpl(self, activity, mode_ref=None):
        if not mode_ref:
            mode_ref = random.choice(MODES_REF)
        lst = []
        act, obj, loc_abs, loc_rel, obj_ref = activity
        t_loc_abs = t_loc_rel = ''
        if loc_abs:
            t_loc_abs = tmpl2txt_loc(loc_abs, loc_rel, obj_ref, LOC_ABS)
            print(t_loc_abs)
        if loc_rel and obj_ref:
            t_loc_rel = tmpl2txt_loc(loc_abs, loc_rel, obj_ref, LOC_REL)
        if act == ADD:
            t_obj = tmpl2txt_obj(obj.color, obj.shape, None, None, None)
        else:
            color, shape, loc_abs_o, loc_rel_o, obj_ref_o = self.canvas.get_features_obj_ref(obj)
            t_obj = tmpl2txt_obj(color, shape, loc_abs_o, loc_rel_o, obj_ref_o)
        message = tmpl2txt_act(act, t_obj, t_loc_abs, t_loc_rel)
        return message

    def get_next_message(self, actions_pre, dial_act_pre, role_pre):
        actions = []
        role = OTHER_ROLE[role_pre]
        return role, actions_pre


if __name__ == '__main__':
    agent = Agent()
    lst_activity = [ADD, DELETE]
    for act in lst_activity:
        act, obj, c, d, e = agent.get_activity(act)
        message = agent.generate_act_message_by_tmpl(act, obj, c, d, e)
        print(act, obj, c, d, e)
        print(message)
