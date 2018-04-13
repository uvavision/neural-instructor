import random
from collections import Counter, defaultdict, OrderedDict
import os
import datetime
import json
from tqdm import tqdm
from utils import get_hashcode
from string import Template
import re

from const import *

# os.environ['domain'] = '2Dshape'
# from main import coll_chat

MODE_LOCS = [LOC_ABS, LOC_REL]

actions = ['add', 'remove', 'move']

abs_loc_dict = {
    'top-left': (0, 0),
    'top-right': (0, 4),
    'bottom-left': (4, 0),
    'bottom-right': (4, 4),
    'center': (2, 2)
}

coord2abs_loc_name = {v: k for k, v in abs_loc_dict.items()}

relative_loc_dict = {
    'top-of': lambda row, col: (row - 1, col),
    'bottom-of': lambda row, col: (row + 1, col),
    'right-of': lambda row, col: (row, col + 1),
    'left-of': lambda row, col: (row, col - 1),
    'top-left-of': lambda row, col: (row - 1, col - 1),
    'top-right-of': lambda row, col: (row - 1, col + 1),
    'bottom-left-of': lambda row, col: (row + 1, col - 1),
    'bottom-right-of': lambda row, col: (row + 1, col + 1)
}


colors = ['red', 'green', 'blue']
shapes = ['square', 'circle', 'triangle']


def xs(o):
    if o is None:
        return ''
    return str(o)


def tmpl2txt_act(act, t_obj, t_loc_abs="", t_loc_rel=""):
    tmpl = random.choice(DICT_TEMPLATES[act])
    s = Template(tmpl)
    text = s.substitute(obj=t_obj, loc_abs=t_loc_abs, rel_loc=t_loc_rel)
    return re.sub(' +', ' ', text)  # remove duplicated spaces


def tmpl2txt_loc(loc_abs, loc_rel, obj_ref, mode_loc=None):
    if not mode_loc:
        mode_loc = random.choice(MODES_LOC)
    tmpl = random.choice(DICT_TEMPLATES[mode_loc])
    s = Template(tmpl)
    text = ''
    if mode_loc == LOC_ABS and loc_abs:
        text = s.substitute(loc_abs)
    if mode_loc == LOC_REL and loc_rel and obj_ref:
        text = s.substitute(loc_rel)
    return text


def tmpl2txt_obj(color, shape, loc_abs, loc_ref, obj_ref, mode_loc=None):
    tmpl = "the $color $shape one $loc_abs $loc_ref"
    s = Template(tmpl)
    text = s.substitute(xs(color), xs(shape))
    text += ' ' + tmpl2txt_loc(loc_abs, loc_rel, obj_ref, mode_loc)
    return re.sub(' +', ' ', text)


def get_rel_loc(obj1, obj2):
    del_c = obj1.col - obj2.col
    del_r = obj1.row - obj2.row
    if (del_c, del_r) in DICT_DEL_REL_LOC:
        return DICT_DEL_REL_LOC[(del_c, del_r)]
    return None


class Object:
    def __init__(self, color, shape, row, col):
        self.color = color
        self.shape = shape
        self.row = row
        self.col = col
        self.identify_status = 'color-shape-location'
        self.id_ = self.get_id()

    def get_id(self):
        id_ = get_hashcode([self.color, self.shape, self.row, self.col])
        return id_

    def get_desc(self):
        assert self.identify_status in ['color-shape', 'location']
        if self.identify_status == 'color-shape':
            return "{} {}".format(self.color, self.shape)
        elif self.identify_status == 'location':
            return "{} {} on {} of the canvas".format(self.color, self.shape, coord2abs_loc_name[self.row, self.col])

    def get_info(self):
        return {'row': self.row, 'col': self.col, 'color': self.color, 'shape': self.shape}


class Canvas:
    def __init__(self, grid_size=GRID_SIZE):
        self.objects = []
        self.color_shape_cnt = Counter()
        self.d_id_obj = {}
        self.d_feature_id = defaultdict(list)  # key (color, shape)
        self.d_id_ref_feature = defaultdict(list)
        self.d_spatical_rel = defaultdict(dict)
        self.d_grid_state = OrderedDict()  # currently not maintain all history
        self.grid_size = grid_size

    def size():
        return len(d_id_obj)

    def get_ref_obj(self):
        # randomly select an object identified by color-shape or abs location as reference
        available = [obj for obj in self.objects if
                     obj.identify_status == 'color-shape' or obj.identify_status == 'location']
        assert len(available) > 0
        return random.choice(available)

    def exec_instruction(self, inst):
        assert inst.viable(self)
        if isinstance(inst, AddInstruction):
            self.objects.append(inst.obj)
        elif isinstance(inst, PatternAddInstruction):
            for obj in inst.objs:
                self.objects.append(obj)
        elif isinstance(inst, RemoveInstruction):
            self.objects.remove(inst.target_obj)
        elif isinstance(inst, MoveInstruction):
            self.objects.remove(inst.remove_inst.target_obj)
            self.objects.append(inst.add_inst.obj)
        self.update_obj_identify_status()

    def update_obj_identify_status(self):
        self.color_shape_cnt = Counter(['{}-{}'.format(obj.color, obj.shape) for obj in self.objects])
        for obj in self.objects:
            if self.color_shape_cnt['{}-{}'.format(obj.color, obj.shape)] == 1:
                obj.identify_status = "color-shape"
            elif (obj.row, obj.col) in abs_loc_dict.values():
                obj.identify_status = "location"
            else:
                obj.identify_status = "color-shape-location"

    def get_desc(self):
        grid_size = 100
        layout = []
        for obj in self.objects:
            top = obj.row * grid_size + 10
            left = obj.col * grid_size + 10
            width = grid_size - 20
            height = grid_size - 20
            label = obj.color
            shape = obj.shape
            if shape == 'square':
                shape = 'rectangle'
            layout.append({"left": left, "top": top, "width": width, "height": height, "label": label, "shape": shape})
        return '#CANVAS-' + str(layout).replace("'", '"').replace(' ', '')

    def get_state_by_coordinates(self, obj):
        # y = obj.row
        # x = obj.col
        # if y in self.d_grid_state and x in in self.d_grid_state[y]:
        #     return self.d_grid_state[y][x]
        if (obj.row, obj.col) in self.d_grid_state:
            return self.d_grid_state[(obj.row, obj.col)]
        return STATE_EMPTY

    def update_spatial_ref(self, obj_new):
        for id_, obj in self.d_id_obj:
            rel = get_rel_loc(obj, obj_new)
            if rel:
                self.d_spatical_rel[obj.id_][obj_new.id_] = rel
            rel = get_rel_loc(obj_new, obj)
            if rel:
                self.d_spatical_rel[obj_new.id_][obj.id_] = rel

    def update_ref_obj_features(self, obj_rm):  # already removed
        if (obj_rm.color, obj_rm.shape) in self.d_feature_id:
            if len(self.d_feature_id) == 1:
                id_ = self.d_feature_id[(obj_rm.color, obj_rm.shape)]
                features = self.get_ref_obj_features(self.d_id_obj[id_])
                self.d_id_ref_feature[id_] = features

    def is_action_viable(self, obj, action):
        if not (0 <= self.obj.row <= self.grid_size and 0 <= self.obj.col <= self.grid_size):
            return False
        if action == ADD:
            state = self.get_state_by_coordinates(obj)
            if state == STATE_OCCU:
                return False
            return True
        if action == DELETE:
            if obj.id_ in self.d_id_obj:
                return True
            return False
        return False

    def add(self, obj):
        # TODO: update "self.objects" if necessary
        if is_action_viable(obj, ADD):
            self.d_id_obj[obj.id_] = obj
            self.d_grid_state[(obj.row, obj.col)] = obj.id_  # STATE_OCCU
            self.d_feature_id[(obj.color, obj.shape)].append(obj.id_)
            self.d_feature_id[obj.color].append(obj.id_)
            self.d_feature_id[obj.shape].append(obj.id_)
            features = self.get_ref_obj_features(obj)
            self.d_id_ref_feature[obj.id_] = features
            self.update_spatial_ref(obj)
            return STATUS_SUCCESSFUL
        else:
            return STATUS_FAILED

    def delete(self, obj):
        # TODO: update "self.objects" if necessary
        if is_action_viable(obj, DELETE):
            del self.d_id_obj[obj.id_]
            del self.d_spatical_rel[obj.id_]
            self. = self.get_ref_obj_features(obj)
            self.d_feature_id[(obj.color, obj.shape)].remove(obj.id_)
            self.d_feature_id[obj.color].remove(obj.id_)
            self.d_feature_id[obj.shape].remove(obj.id_)
            self.update_ref_obj_features(obj)
            for k, v in self.d_spatical_rel.items():
                if obj.id_ in v:
                    del self.d_spatical_rel[k][obj.id_]
            self.d_grid_state[(obj.row, obj.col)] = STATE_EMPTY
            return STATUS_SUCCESSFUL
        return STATUS_FAILED

    def move(self, obj_from, obj_to):
        if self.is_action_valid(obj_from, DELETE) and \
           self.is_action_valid(obj_to, ADD):
            return self.delete(obj_from) and self.add(obj_add)
        return STATUS_FAILED

    def get_min_features_obj_ref(self, obj, mode_ref=None):
        lst = []
        color = shape = loc_abs = loc_rel = obj_ref = None
        if obj.color in self.d_feature_id and len(self.d_feature_id[obj.color]) == 1:
            lst.append(obj.color)
        if obj.shape in self.d_feature_id and len(self.d_feature_id[obj.shape]) == 1:
            lst.append(obj.shape)
        if (obj.row, obj.col) in DICT_LOC_ABS2NAME:
            lst.append(DICT_LOC_ABS2NAME(obj.row, obj.col))
        if obj.id_ in d_spatical_rel:
            obj_ref_id, loc_rel = random.choice(list(d_spatical_rel[obj.id_].items()))
            obj_ref = self.d_id_obj[obj_ref_id]
            lst.append((loc_ref, obj_ref)
        return lst

    def get_next_action(self):
        # TODO: get next action based on latest painting
        # e.g., if previous painting is correct, check if complete or add new.
        # e.g., if previous painting is incorrect, correct it.
        raise NotImplementedError


class Instruction:
    def __init__(self, ref_obj, relative_loc, abs_loc, canvas=None):
        # abs_loc: top-left ...
        # relative_loc: top-of ...
        if ref_obj:
            assert relative_loc
            assert not abs_loc
        if abs_loc:
            assert not ref_obj
            assert not relative_loc
        self.ref_obj = ref_obj
        self.relative_loc = relative_loc
        self.absolute_loc = abs_loc
        self.canvas = canvas

    def viable(self, canvas):
        raise NotImplementedError

    def get_loc_desc(self):
        if self.absolute_loc:
            return self.absolute_loc + " of the canvas"
        if self.relative_loc:
            return "{} the {}".format(self.relative_loc, self.ref_obj.get_desc())

    def get_obj_ref_text(self, obj, mode=MODE_FULL):
        if mode == MODE_MIN:
            features = self.canvas.d_id_ref_feature[obj.id_]
            if len(features) > 0:
                return "the {} object".format(random.choice(features))
        return "{} {}".format(obj.color, obj.shape)


class AddInstruction(Instruction):
    def __init__(self, ref_obj, relative_loc, abs_loc, new_obj):
        super().__init__(ref_obj, relative_loc, abs_loc)
        self.obj = new_obj

    def viable(self, canvas):
        # the position to put the new object should be empty
        a = (self.obj.row, self.obj.col) not in [(obj.row, obj.col) for obj in canvas.objects]
        b = (0 <= self.obj.row <= 4 and 0 <= self.obj.col <= 4)
        return a and b

    def get_desc(self):
        # adding objects always requires specifying the location
        return "add a {} {} to {}".format(self.obj.color, self.obj.shape, self.get_loc_desc())


class PatternAddInstruction(Instruction):
    def __init__(self, ref_obj, relative_loc, abs_loc, new_objs):
        super().__init__(ref_obj, relative_loc, abs_loc)
        assert len(new_objs) == 3
        rows = [obj.row for obj in new_objs]
        cols = [obj.col for obj in new_objs]
        obj_colors = [obj.color for obj in new_objs]
        obj_shapes = [obj.shape for obj in new_objs]
        assert obj_colors[0] == obj_colors[1] == obj_colors[2]
        assert obj_shapes[0] == obj_shapes[1] == obj_shapes[2]
        if rows[0] == rows[1] == rows[2]:
            self.style = "row"
            assert cols[0] + 1 == cols[1] and cols[1] + 1 == cols[2]
        elif cols[0] == cols[1] == cols[2]:
            self.style = 'column'
            assert rows[0] + 1 == rows[1] and rows[1] + 1 == rows[2]
        else:
            assert False
        assert self.style in ['row', 'column']
        self.objs = new_objs

    def viable(self, canvas):
        # the position to put the new object should be empty
        invalid_pos = [(obj.row, obj.col) for obj in canvas.objects]
        for obj in self.objs:
            if obj.row < 0 or obj.row > 4 or obj.col < 0 or obj.col > 4:
                return False
            if (obj.row, obj.col) in invalid_pos:
                return False
        return True

    def get_desc(self):
        # adding objects always requires specifying the location
        return "add a {} of three {} {}s to {}".format(self.style, self.objs[0].color, self.objs[0].shape, self.get_loc_desc())


class RemoveInstruction(Instruction):
    def __init__(self, ref_obj, relative_loc, abs_loc, target_obj):
        super().__init__(ref_obj, relative_loc, abs_loc)
        assert target_obj
        self.target_obj = target_obj

    def viable(self, canvas):
        # the object to be removed must be on the canvas
        a = self.target_obj in canvas.objects
        b = (0 <= self.target_obj.row <= 4 and 0 <= self.target_obj.col <= 4)
        return a and b

    def get_desc(self):
        if self.target_obj.identify_status == 'color-shape':
            return "remove the {} {}".format(self.target_obj.color, self.target_obj.shape)
        else:
            return "remove the {} {} on {}".format(self.target_obj.color, self.target_obj.shape, self.get_loc_desc())


class MoveInstruction:
    def __init__(self, remove_inst, add_inst):
        # move instruction is a remove instruction following an add instruction
        # same color and shape, different position
        obj1 = remove_inst.target_obj
        obj2 = add_inst.obj
        assert (obj1.row, obj1.col) != (obj2.row, obj2.col) and \
               obj1.color == obj2.color and obj1.shape == obj2.shape
        self.remove_inst = remove_inst
        self.add_inst = add_inst

    def viable(self, canvas):
        return self.remove_inst.viable(canvas) and self.add_inst.viable(canvas)

    def get_desc(self):
        return "{} to {}".format(self.remove_inst.get_desc().replace("remove", "move"),
                                 self.add_inst.get_loc_desc())


def get_action_loc_abs(abs_loc):
    action_row, action_col = abs_loc_dict[abs_loc]
    return action_row, action_col


def get_action_loc_relative(ref_obj, relative_loc):
    action_row, action_col = relative_loc_dict[relative_loc](ref_obj.row, ref_obj.col)
    return action_row, action_col


def sample_abs_loc(canvas):
    abs_loc = random.choice(list(abs_loc_dict.keys()))
    action_row, action_col = get_action_loc_abs(abs_loc)
    return abs_loc, action_row, action_col


def sample_relative_loc(canvas):
    ref_obj = canvas.get_ref_obj()
    relative_loc = random.choice(list(relative_loc_dict.keys()))
    action_row, action_col = get_action_loc_relative(ref_obj, relative_loc)
    return ref_obj, relative_loc, action_row, action_col


def get_add_inst(canvas, single_obj_only=False):
    single_obj_only = True
    while True:
        loc = 'abs' if len(canvas.objects) == 0 else random.choice(['abs', 'relative'])
        abs_loc = ref_obj = relative_loc = None
        if loc == 'abs':
            abs_loc, action_row, action_col = sample_abs_loc(canvas)
        if loc == 'relative':
            ref_obj, relative_loc, action_row, action_col = sample_relative_loc(canvas)

        if single_obj_only or random.random() < 0.5:
            new_obj = Object(random.choice(colors), random.choice(shapes), action_row, action_col)
            inst = AddInstruction(ref_obj, relative_loc, abs_loc, new_obj)
            if inst.viable(canvas):
                return inst
        else:
            pattern_color = random.choice(colors)
            pattern_shape = random.choice(shapes)
            style = random.choice(['row', 'column'])
            if style == 'row':
                obj1 = Object(pattern_color, pattern_shape, action_row, action_col - 1)
                obj2 = Object(pattern_color, pattern_shape, action_row, action_col)
                obj3 = Object(pattern_color, pattern_shape, action_row, action_col + 1)
            if style == 'column':
                obj1 = Object(pattern_color, pattern_shape, action_row - 1, action_col)
                obj2 = Object(pattern_color, pattern_shape, action_row, action_col)
                obj3 = Object(pattern_color, pattern_shape, action_row + 1, action_col)
            inst = PatternAddInstruction(ref_obj, relative_loc, abs_loc, [obj1, obj2, obj3])
            if inst.viable(canvas):
                return inst


def get_remove_inst(canvas):
    assert len(canvas.objects) > 0
    while True:
        loc = 'abs' if len(canvas.objects) == 1 else random.choice(['abs', 'relative'])
        abs_loc = ref_obj = relative_loc = None
        if loc == 'abs':
            abs_loc, action_row, action_col = sample_abs_loc(canvas)
        if loc == 'relative':
            ref_obj, relative_loc, action_row, action_col = sample_relative_loc(canvas)
        target_obj = None
        for obj in canvas.objects:
            if (action_row, action_col) == (obj.row, obj.col):
                target_obj = obj
        # if target obj is not valid or target obj is the same with reference obj
        if not target_obj or (ref_obj and target_obj and ref_obj == target_obj):
            continue
        inst = RemoveInstruction(ref_obj, relative_loc, abs_loc, target_obj)
        if inst.viable(canvas):
            return inst


def get_move_inst(canvas):
    assert len(canvas.objects) > 0
    while True:
        remove_inst = get_remove_inst(canvas)
        add_inst = get_add_inst(canvas, single_obj_only=True)
        # object to be removed cannot be used as reference
        while remove_inst.target_obj == add_inst.ref_obj:
            add_inst = get_add_inst(canvas, single_obj_only=True)

        add_inst.obj.color = remove_inst.target_obj.color
        add_inst.obj.shape = remove_inst.target_obj.shape
        move_inst = MoveInstruction(remove_inst, add_inst)
        if move_inst.viable(canvas):
            return move_inst


def construct_next_instruction(canvas):
    weighted_actions = ['add'] * 5 + ['move'] * 3 + ['remove'] * 1
    action = 'add' if len(canvas.objects) == 0 else random.choice(weighted_actions)
    action = 'add'
    return {'add': get_add_inst, 'remove': get_remove_inst, 'move': get_move_inst}[action](canvas)


def get_add_activity(canvas, mode_loc=None, mode_style=SINGLE):
    if canvas.size() == 0:
        mode_loc = LOC_ABS
    if mode_loc is None:
        mode_loc = random.choice(MODES_LOC)
    obj_new = obj_ref = loc_abs = loc_rel = None
    if mode_loc == LOC_ABS:
        abs_loc, row, col = sample_abs_loc(canvas)
    if mode_loc == LOC_REL:
        obj_rel, loc_rel, row, col = sample_relative_loc(canvas)
    if mode_style == SINGLE:
        obj_new = Object(random.choice(COLORS), random.choice(SHAPES), action_row, action_col)
        return (obj_new, loc_abs, obj_ref, loc_rel)
    elif mode_style == PATTERN:
        color = random.choice(COLORS)
        shape = random.choice(SHAPES)
        style = random.choice(PATTERN_STYLE)
        if style == 'row':
            obj1 = Object(color, shape, row, col - 1)
            obj2 = Object(color, shape, row, col)
            obj3 = Object(color, shape, row, col + 1)
        if style == 'column':
            obj1 = Object(color, shape, row - 1, col)
            obj2 = Object(color, shape, row, col)
            obj3 = Object(color, shape, row + 1, col)
        return ([obj1, obj2, obj3], loc_abs, obj_ref, loc_rel)
    return None


def get_delete_activity(canvas, mode_loc=None, mode_style=SINGLE):
    assert len(canvas.objects) > 0
    if canvas.size() == 1:
        mode_loc = LOC_ABS
    if not mode_loc:
        mode_loc = random.choice(MODES_LOC)
    obj_del = obj_rel = loc_abs = loc_rel = None
    if mode_loc == LOC_ABS:
        loc_abs, row, col = sample_abs_loc(canvas)
    if mode_loc == LOC_REL:
        obj_rel, loc_rel, row, col = sample_relative_loc(canvas)
    for obj in canvas.objects:
        if (row, col) == (obj.row, obj.col):
            obj_del = obj
    # if target obj is not valid or target obj is the same with reference obj
        if not obj_del or (obj_rel and obj_del and obj_rel == obj_del):
            continue
        return (obj_del, loc_abs, obj_ref, loc_rel)
    return None


def get_move_activity(canvas, mode_loc=None, mode_style=SINGLE):
    assert len(canvas.objects) > 0
    del_activity = get_delete_activity(canvas, mode_loc)
    obj_from = del_activity[0]
    add_activity = get_add_activity(canvas, mode_style, mode_loc)
    obj_to = add_activity[0]
    while obj_from.row == obj_to.row and obj_from.col == obj_from.col:
        add_activity = get_add_activity(canvas, mode_style, mode_loc)
        obj_to = add_activity[0]
    add_activity[0] = Object(obj_from.color, obj_from.shape, obj_to.col, obj_to.row)
    return del_activity, add_activity


if __name__ == '__main__':
    train_set_size = 100000
    train_set = []
    for ix in tqdm(range(train_set_size)):
        data = []
        canvas = Canvas()
        for i in range(10):
            inst = construct_next_instruction(canvas)
            prev_canvas = [o.get_info() for o in canvas.objects]
            current_instuction = inst.get_desc()
            next_obj = inst.obj.get_info()
            if inst.ref_obj:
                # if 'of the canvas' in current_instuction:
                data.append({"prev_canvas": prev_canvas,
                             "current_instruction": current_instuction,
                             "next_object": next_obj,
                             "ref_obj": inst.ref_obj.get_info()})
            else:
                data.append({"prev_canvas": prev_canvas,
                             "current_instruction": current_instuction,
                             "next_object": next_obj})

            canvas.exec_instruction(inst)
        for item in data:
            item['final_canvas'] = [o.get_info() for o in canvas.objects]
        train_set.extend(data)
    print(json.dumps(train_set))
    # print(json.dumps(train_set, indent=2))
    # print(json.dumps(train_set, indent=2, sort_keys=True))
    # task_id = '12555'
    # coll_chat.delete_many({'task_id': task_id})
    # canvas = Canvas()
    # for i in range(50):
    #     inst = construct_next_instruction(canvas)
    #     print("{}: {}".format(i, inst.get_desc()))
    #     coll_chat.insert({'msg': inst.get_desc(), 'author': 'human', 'task_id': task_id,
    #                       'username': 'aaa', 'worker_id': '123456', "role": 'user', 'turn': str(i), 'mode': '2Dshape',
    #                       'timestamp': str(datetime.datetime.now())})
    #     canvas.exec_instruction(inst)
    #     coll_chat.insert({'msg': canvas.get_desc(), 'author': 'human', 'task_id': task_id,
    #                       'username': 'bbb', 'worker_id': '123456', "role": 'agent', 'turn': str(i), 'mode': '2Dshape',
    #                       'timestamp': str(datetime.datetime.now())})
