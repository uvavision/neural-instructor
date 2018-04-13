from const import *
from collections import OrderedDict
from synthetic_layout import (Object, Canvas, get_add_activity,
                              get_delete_activity, get_move_activity,
                              tmpl2txt_act, tmpl2txt_loc)


class Policy(object):
    def __init__(self, domain, path):
        self.domain = domain
        self.policy = load(path)

    def load(self, path=None):
        d = {}
        return d


class Agent(object):

    def __init__(self, policy, domain=DOMAIN_2DSHAPE):
        self.domain = domain
        self.policy = policy
        self.states = OrderedDict
        self.canvas = Canvas()

    def get_act_obj(self, actions=[ADD], mode_loc=None, mode_style=None):
        if not mode_style:
            mode_style = random.choice(MODES_STYLE)
        if not mode_loc:
            mode_loc = random.choice(MODES_LOC)
        lst_activity = []
        for action in actions:
            if action == ADD:
                activity = get_add_activity(self.canvas, mode_loc, mode_style)
            if action == DELETE:
                activity = get_delete_activity(self.canvas, mode_loc, mode_style)
            if action == MOVE:
                activity = get_move_activity(self.canvas, mode_loc, mode_style)
            if activity:
                lst_activity.append((action,) + activity)
        return lst_activity

    def generate_act_message_by_tmpl(self, lst_activity, mode_ref=None):
        if not mode_ref:
            mode_ref = random.choice(MODES_REF)
        lst = []
        for act, obj, loc_abs, loc_rel, obj_ref in lst_activity:
            t_loc_abs = t_loc_rel = ''
            if loc_abs:
                t_loc_abs = tmpl2txt_loc(loc_abs, loc_rel, obj_ref, LOC_ABS)
            if loc_rel and obj_ref:
                t_loc_rel = tmpl2txt_loc(loc_abs, loc_rel, obj_ref, LOC_REF)
            color, shape, loc_abs, loc_ref, obj_ref = canvas.?
            t_obj = tmpl2txt_obj(color, shape, loc_abs, loc_ref, obj_ref)
            message = tmpl2txt_act(ADD, t_obj, t_loc_abs, t_loc_rel)

    def get_next_message(self, actions_pre, dial_act_pre, role_pre):
        actions = []
        role = OTHER_ROLE[role_pre]
        return role, actions_pre


if __name__ == '__main__':
    pass
