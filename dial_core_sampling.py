import sys

from const import *
from utils import alter
import random
import json
from collections import OrderedDict, Counter
from layout import (Object, Canvas, tmpl2txt_act, tmpl2txt_da)
from pprint import pprint
import requests
import copy
from data_utils import *
from tqdm import tqdm


class Policy(object):
    # TODO: load policy from domain config file
    def __init__(self, domain, path):
        self.domain = domain
        self.policy = self.load(path)

    def random_gen_policy(self):
        pass

    def load_policy(self, path=None):
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

    def reset_activity(self):
        self.act = None
        self.obj = None
        self.loc_abs = None
        self.loc_rel = None
        self.obj_ref = None
        self.message = None

    def reset_config(self, mode_loc=None, mode_ref=None):
        self.mode_loc = mode_loc
        self.mode_ref = mode_ref

    def activity2dict(self):
        d = {ACT: self.act,
             OBJ: self.obj.to_dict() if self.obj else None,
             # LOC_ABS: self.loc_abs,
             # LOC_REL: self.loc_rel,
             # OBJ_REF: self.obj_ref.to_dict() if self.obj_ref else None,
             FEATURES: self.obj.features,
             MESSAGE: self.message
        }
        return d

    def config2dict(self):
        d = {'mode_loc': self.mode_loc, 'mode_ref': self.mode_ref}
        return d

    def get_add_activity(self, select_empty=True, excluded_grids=[]):
        if self.canvas.size() == 0:
            self.mode_loc == LOC_ABS
        options = []
        row_abs, col_abs, loc_abs = self.canvas.select_grid_loc_abs(select_empty)
        if row_abs is not None and col_abs is not None:
            options.append(LOC_ABS)
        row_ref, col_ref, obj_ref, loc_rel = self.canvas.select_grid_obj_ref(select_empty, excluded_grids)
        if obj_ref:
            options.append(LOC_REL)
        if len(options) == 0:
            return
        color = random.choice(COLORS)
        shape = random.choice(SHAPES)
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

    def get_delete_activity(self, select_empty=False):
        assert self.canvas.size() > 0
        if self.canvas.size() == 1 and not select_empty:
            self.obj = list(self.canvas.d_id_obj.values())[0]
            return
        row = col = None
        row_a, col_a, self.loc_abs = self.canvas.select_grid_loc_abs(select_empty)
        row_r, col_r, self.obj_ref, self.loc_rel = self.canvas.select_grid_obj_ref(select_empty)
        options = []
        if row_a is not None:
            options.append(LOC_ABS)
        if row_r is not None:
            options.append(LOC_REL)
        if not options:
            for id_, obj in self.canvas.d_id_obj.items():
                d_features = self.canvas.get_obj_features(obj)
                if MODE_CS in d_features:
                    self.obj = obj
                    break
            return
        if self.mode_loc not in options:
            self.mode_loc = random.choice(options)
        if row_r is not None and self.mode_loc == LOC_REL:
            row, col = row_r, col_r
        if row_a is not None and self.mode_loc == LOC_ABS:
            row, col = row_a, col_a
        if row is not None and col is not None:
            if select_empty:
                color = random.choice(COLORS)
                shape = random.choice(SHAPES)
                self.obj = Object(color, shape, row, col)
            else:
                for id_, obj in self.canvas.d_id_obj.items():
                    if (row, col) == (obj.row, obj.col):
                        self.obj = obj
                        break

    def get_activities(self, act, is_viable=True):
        if not self.mode_loc:
            self.mode_loc = random.choice(MODES_LOC)
        activities = []
        if act == ADD:
            self.act = act
            self.get_add_activity(select_empty=(True and is_viable))
            self.message = generate_act_message_by_tmpl(self.canvas, self.obj, self.act, MODE_FULL)
            self.canvas.add(self.obj)
            activities.append(self.activity2dict())
        if act == DELETE:
            self.act = act
            self.get_delete_activity(select_empty=not(False ^ is_viable))
            if self.obj is None:
                return []
            # print(self.obj)
            if not self.mode_ref:
                mode_ref = random.choice(MODES_REF)
            else:
                mode_ref = self.mode_ref
            # print(self.obj)
            self.message = generate_act_message_by_tmpl(self.canvas, self.obj, self.act, mode_ref)
            self.canvas.delete(self.obj)
            activities.append(self.activity2dict())
        if act == MOVE:
            self.get_delete_activity(select_empty=False)
            self.act = DELETE
            activities.append(self.activity2dict())
            obj_from = self.obj
            self.get_add_activity(select_empty=True, excluded_grids=[(obj_from.row, obj_from.col)])
            self.act = ADD
            self.obj.color, self.obj.shape = obj_from.color, obj_from.shape
            obj_to = Object(self.obj.color, self.obj.shape, self.obj.row, self.obj.col)
            self.obj = obj_to
            activities.append(self.activity2dict())
            self.obj = obj_from
            # self.message = self.generate_act_message_by_tmpl(MOVE)
            self.canvas.delete(obj_from)
            self.canvas.add(obj_to)
        self.canvas.update_ref_obj_features()
        # for k, v in self.canvas.d_id_feature.items():
        #     print(k, v.keys())
        return activities


def generate_act_message_by_tmpl(canvas, obj, act, mode_ref, role=USER, da=REQUEST):
    t_loc_abs = t_loc_rel = ''
    t_obj = canvas.get_obj_desc(obj, mode_ref, mode_ref)
    # if act == MOVE:
    #     t_obj = canvas.get_obj_desc(obj, mode_ref, mode_ref)
    #     if obj.loc_abs:
    #         t_loc_abs = canvas.get_loc_desc(obj.loc_abs, obj.loc_rel, obj.obj_ref, mode_ref)
    #     if obj.loc_rel and obj.obj_ref:
    #         t_loc_rel = canvas.get_loc_desc(obj.loc_abs, obj.loc_rel, obj.obj_ref, mode_ref)
    message = tmpl2txt_act(act, da, role, t_obj, t_loc_abs, t_loc_rel)
    return message


def get_act_sequence(n_obj):
    lst_act = [ADD]
    n_delete = random.randint(1, 3)
    n_add = n_obj + n_delete
    n_move = 0 # random.randint(1, 2)
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


def get_altered_activity(agent, d_activity, alter_type, is_viable):
    act = agent.act
    features_altered = []
    d_activity_alter = d_activity # copy.deepcopy(d_activity)
    if alter_type is None:
        alter_type = random.sample([1, 2, 3])
    if alter_type == 1 or 1 in alter_type:  # change color
        color = alter(d_activity[OBJ][COLOR], COLORS)
        d_activity_alter[OBJ].update({COLOR: color, ID:None, FEATURES: None})
        features_altered.append(COLOR)
    if alter_type == 2 or 2 in alter_type:  # change shape
        shape = alter(d_activity[OBJ][SHAPE], SHAPES)
        d_activity_alter[OBJ].update({SHAPE: shape, ID: None, FEATURES: None})
        features_altered.append(SHAPE)
    if alter_type == 3 or 3 in alter_type:  # change selected grid
        row = d_activity[OBJ][ROW]
        col = d_activity[OBJ][COL]
        select_empty = None
        if act == ADD:
            select_empty = True and is_viable
        if act == DELETE:
            select_empty = not(False ^ is_viable)
        row_abs, col_abs, loc_abs = agent.canvas.select_grid_loc_abs(select_empty=select_empty, excluded_grids=[(row, col)])
        row_rel, col_rel, obj_ref, loc_rel = agent.canvas.select_grid_obj_ref(select_empty=select_empty,
                                                                              excluded_grids=[(row, col)])
        r1 = row_abs is not None
        r2 = row_rel is not None
        random_selector = random.uniform(0, 1)
        if (r1 and r2 and random_selector >= 0.5) or r1:
            d_activity_alter[OBJ].update({ROW: row_abs, COL: col_abs, LOC_ABS: loc_abs})
            agent.loc_abs = loc_abs
            features_altered.append(LOC_ABS)
        elif (r1 and r2 and random_selector < 0.5) or r2:
            d_activity_alter[OBJ].update({ROW: row_rel, COL: col_rel, LOC_REL: loc_rel, OBJ_REF: obj_ref.to_dict()})
            agent.loc_rel = loc_rel
            agent.obj_ref = obj_ref
            features_altered.extend(LOC_REL)
    if alter_type == 4:  # todo: wrong act
        pass
    agent.obj.set(d_activity_alter[OBJ])
    return d_activity_alter, features_altered


def get_next_turn(agent, d_activity_pre=None, pre_role=USER, pre_da=REQUEST, is_viable_pre=True):
    d_para = {}
    if d_activity_pre is None or (pre_role, pre_da, True) not in D_ACTION_MAPPING:
        return USER, REQUEST, d_para, None
    role, da, is_viable = random.choice(D_ACTION_MAPPING[(pre_role, pre_da, is_viable_pre)])
    if role == USER and da == SELF_CORRECTION:
        if is_viable:
            n = random.randint(1, 2)
            alter_type = random.sample([1, 2], n)
        else:
            # n = random.randint(0, 2)
            # alter_type = random.sample([1, 2], n) + [3]
            alter_type = [3]
        d_activity_alter, features_altered = get_altered_activity(agent, d_activity_pre, alter_type, not is_viable)
        if agent.act == ADD:
            agent.canvas.add(agent.obj)
        if agent.act == DELETE:
            agent.canvas.delete(agent.obj)
        for feature in features_altered:
            d_para[feature] = d_activity_alter[OBJ][feature]
    if role == AGENT and da == ASK_Q:
        obj = agent.obj
        loc_abs = agent.loc_abs
        loc_rel = agent.loc_rel
        obj_ref = agent.obj_ref
        d_para[OBJ] = agent.canvas.get_obj_desc(obj, None, None, [(obj.color, obj.shape, None, None, None)])
        d_para[T_LOC_ABS] = agent.canvas.get_loc_desc(loc_abs, None, None)
        d_para[T_LOC_REL] = agent.canvas.get_loc_desc(None, loc_rel, obj_ref)
    return role, da, d_para, is_viable


def grid_maker(d_id_obj, h=GRID_SIZE, w=GRID_SIZE):
    grid = [["[  ]" for i in range(w)] for i in range(h)]
    for id_, obj in d_id_obj.items():
        grid[obj.row][obj.col] = obj.color[0] + '/' + obj.shape[0]
    return grid

def obj2grid(objs, h=GRID_SIZE, w=GRID_SIZE):
    grid = [["[  ]" for i in range(w)] for i in range(h)]
    for obj in objs:
        grid[obj['row']][obj['col']] = obj['color'][0] + '|' + obj['shape'][0]
    return grid


def generate_data(n_dial, is_viable=True, mode_ref=None, out_json=None):
# def generate_data(n_dial, is_viable=True, mode_ref=MODE_MIN, out_json='data.json'):
    data = []
    with_delete_count = 0
    # for i in range(n_dial):
    for i in tqdm(range(n_dial)):
        d_dial = {'dial_id': i + 1, 'dialog_data': []}
        n_obj = random.randint(3, 6)
        lst_act = get_act_sequence(n_obj)
        while 'move' in lst_act:
            lst_act.remove('move')
        assert not 'move' in lst_act
        verbose = False
        if verbose:
            print(lst_act)
        is_discard = False
        d_id_act = OrderedDict()
        agent = Agent(mode_loc=None, mode_ref=mode_ref)
        detailed_ref_types = []
        ref_types = []
        generate_dialog_slice_data = True
        slice_valid = False
        for turn, act in enumerate(lst_act):
            turn = turn + 1
            prev_canvas = [ele.to_dict() for ele in agent.canvas.d_id_obj.values()]
            activities = agent.get_activities(act)
            # obj_rel = activities[0]['obj_rel']
            # if obj_rel is not None:
            #     feature_obj_rel = activities[0]['features']['obj_rel']
            #     # if feature_obj_rel is not None:
            #     a = '{}-{}-{}-{}'.format(obj_rel['color'], obj_rel['shape'], obj_rel['row'], obj_rel['col'])
            #     b = '{}-{}-{}-{}'.format(feature_obj_rel['color'], feature_obj_rel['shape'], feature_obj_rel['row'], feature_obj_rel['col'])
            #     assert a == b
            if activities is None or activities == []:
                is_discard = True
                break
            current_canvas = [ele.to_dict() for ele in agent.canvas.d_id_obj.values()]
            for d_activity in activities:
                d_id_act.setdefault(d_activity[OBJ][ID], []).append(d_activity[ACT])
                d = {'turn': turn, 'config': agent.config2dict(), 'activities': activities,
                     'prev_canvas': prev_canvas, 'current_canvas': current_canvas}
                d_dial['dialog_data'].append(d)
            if verbose:
                grid = grid_maker(agent.canvas.d_id_obj)
                print("#", agent.act, '#', agent.message)
                pprint(grid)
            # # print(agent.canvas.get_desc())
            agent.reset_activity()
            agent.reset_config(mode_loc=None, mode_ref=mode_ref)
            ref_type = inst_ref_type(activities[0]['message'])
            detailed_ref_types.append(ref_type)
            # if ref_type != INST_REL:
            #     activities[0]['features']['obj_ref'] = None
            if ref_type == INST_ABS:
                ref_types.append("abs")
            elif ref_type == INST_REL or ref_type == INST_REL_ABS:
                ref_types.append("rel")
            else:
                # don't consider INST_REL_REL_ABS
                ref_types.append("invalid")
            if generate_dialog_slice_data and turn >= 2 + random.randint(1, 5):
                slice_valid = True
                break
                # if ref_types[-3:] == ['abs', 'rel', 'rel']:
                #     # acts = [d['activities'][0]['act'] for d in d_dial['dialog_data'][-3:]]
                #     # if 'delete' in acts:
                #     # TODO
                #     slice_valid = True
                #     break
                #     # if Counter(acts)['delete'] == 1:
                #     #     with_delete_count += 1
                #     #     break
        if is_discard:
            continue
        for k, v in d_id_act.items():
            if v not in ([ADD], [ADD, DELETE]):
                is_discard = True
                break
        if is_viable and is_discard:
            continue
        if 'invalid' in ref_types:
            continue
        if not generate_dialog_slice_data:
            data.append(d_dial)
            continue
        #  generate dialog slice data
        if slice_valid:
            targets = []
            d_dial['dialog_data'] = d_dial['dialog_data'][-3:]
            msgs = [d['activities'][0]['message'] for d in d_dial['dialog_data']]
            for d_ix in range(len(d_dial['dialog_data'])):
                ddata = d_dial['dialog_data'][d_ix]
                # if ddata['activities'][0]['act'] == 'delete' and \
                #         detailed_ref_types[-3:][d_ix] == INST_REL_ABS:
                #     msg = ddata['activities'][0]['message']
                #     if 'object' not in msg.split():
                #         valid = False
                target = ddata['activities'][0]['obj']
                targets.append('{}-{}-{}-{}'.format(target['color'], target['shape'], target['row'], target['col']))
                prev_canvas = obj2grid(ddata['prev_canvas'])
                if verbose:
                    pprint(prev_canvas)
                    print(ddata['activities'][0]['message'])
            if verbose:
                pprint(obj2grid(d_dial['dialog_data'][-1]['current_canvas']))
                print('=========================================================')
            if len(set(targets)) != len(targets):
                slice_valid = False
            if slice_valid:
                acts = [d['activities'][0]['act'] for d in d_dial['dialog_data']]
                if Counter(acts)['delete'] > 0:
                    with_delete_count += 1
                # d_dial['final_layout'] = [v.to_dict() for k, v in agent.canvas.d_id_obj.items()]
                data.append(d_dial)

    if out_json:
        print("number of dialogs: {}, with delete: {}".format(len(data), with_delete_count))
        json.dump(data, open(out_json, 'w'), indent=4)


def get_para_act(agent, is_viable=True, act=None, obj=None, mode_ref=None):
    canvas = agent.canvas
    act = agent.act if act is None else act
    obj = agent.obj if obj is None else obj
    t_obj = t_loc_abs = t_loc_rel = ''
    if act == ADD:
        t_obj = canvas.get_obj_desc(obj, MODE_FULL, mode_ref)
    if act == DELETE:
        if not is_viable:
            t_obj = canvas.get_obj_desc(obj, MODE_FULL, mode_ref)
        else:
            t_obj = canvas.get_obj_desc(obj, mode_ref, mode_ref)
    if act == MOVE:
        t_obj = canvas.get_obj_desc(obj, mode_ref, mode_ref)
        if obj.loc_abs:
            t_loc_abs = canvas.get_loc_desc(obj.loc_abs, obj.loc_rel, obj.obj_ref, mode_ref)
        if obj.loc_rel and obj.obj_ref:
            t_loc_rel = canvas.get_loc_desc(obj.loc_abs, obj.loc_rel, obj.obj_ref, mode_ref)
    return {ACT: act, T_OBJ: t_obj, T_LOC_ABS: t_loc_abs, T_LOC_REL: t_loc_rel}


def gen_dialog_flow():
    n_obj = random.randint(3, 6)
    lst_act = get_act_sequence(n_obj)
    print('ACTION SEQUENCE', lst_act)
    lst_act.reverse()
    is_viable = act = d_activity_pre = None
    agent = Agent()
    role, da, d_para, _ = get_next_turn(agent, None, None, None)
    while True:
        if role == USER and da == REQUEST:
            act = lst_act.pop()
            if act == ADD and agent.canvas.size() == 0:
                is_viable = True
            elif act == DELETE and agent.canvas.size() == 0:
                is_viable = False
            else:
                is_viable = random.choice([True, False])
            if act == DELETE:
                is_viable = False
            d_activity = agent.get_activities(act, is_viable)[0]
            d_activity[IS_VIABLE] = is_viable
            d_activity_pre = d_activity
            d_para = get_para_act(agent, is_viable)
        text = tmpl2txt_da(role, da, act, d_para, is_viable)
        print('{} | {} | viable:{} || {}'.format(role[0].upper(), da, is_viable, text))
        grid = grid_maker(agent.canvas.d_id_obj)
        pprint(grid)
        role_pre, da_pre = role, da
        if len(lst_act) > 0:
            role, da, d_para, is_viable = get_next_turn(agent, d_activity_pre, role_pre, da_pre, is_viable)
        else:
            if role == USER and da == REQUEST:
                text = tmpl2txt_da(AGENT, CONFIRM, None, None)
                print('{} | {} || {}'.format(AGENT[0].upper(), CONFIRM, text))
            text = tmpl2txt_da(USER, END, None, None)
            print('{} | {} || {}'.format(USER[0].upper(), END, text))
            print()
            break
    grid = grid_maker(agent.canvas.d_id_obj)
    pprint(grid)


def debug():
    agent = Agent(mode_loc=None, mode_ref=None)
    activities = agent.get_activities(ADD)
    grid = grid_maker(agent.canvas.d_id_obj)
    print("#", agent.act, '#', agent.message)
    pprint(grid)
    agent.reset_activity()
    agent.mode_loc = LOC_REL
    activities = agent.get_activities(ADD)
    grid = grid_maker(agent.canvas.d_id_obj)
    print("#", agent.act, '#', agent.message)
    pprint(grid)
    agent.mode_ref = LOC_REL
    agent.mode_loc = LOC_REL
    activities = agent.get_activities(DELETE)
    grid = grid_maker(agent.canvas.d_id_obj)
    print("#", agent.act, '#', agent.message)
    pprint(grid)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        # generate_data(5000000, out_json=sys.argv[1])
        # generate_data(200, out_json=sys.argv[1])
        generate_data(500000//3, out_json=sys.argv[1])
        # generate_data(10000000, out_json=sys.argv[1])
    else:
        generate_data(100000)
