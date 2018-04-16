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
    text = s.substitute(obj=t_obj, loc_abs=t_loc_abs, loc_rel=t_loc_rel)
    return re.sub(' +', ' ', text)  # remove duplicated spaces


def tmpl2txt_loc(loc_abs, loc_rel, obj_ref, mode_loc=None):
    if not mode_loc:
        mode_loc = random.choice(MODES_LOC)
    tmpl = random.choice(DICT_TEMPLATES[mode_loc])
    s = Template(tmpl)
    text = ''
    if mode_loc == LOC_ABS and loc_abs:
        text = s.substitute(loc_abs=loc_abs)
    if mode_loc == LOC_REL and loc_rel and obj_ref:
        t_obj_ref = tmpl2txt_obj(obj_ref)
        text = s.substitute(loc_rel=loc_rel, obj_ref=t_obj_ref)
    return text


def tmpl2txt_obj(color, shape, loc_abs, loc_rel, obj_ref, mode_loc=None):
    tmpl = random.choice(DICT_TEMPLATES[OBJ])
    s = Template(tmpl)
    text = s.substitute(color=xs(color), shape=xs(shape))
    text += ' ' + tmpl2txt_loc(loc_abs, loc_rel, obj_ref, mode_loc)
    return re.sub(' +', ' ', text)


def get_rel_loc(obj1, obj2):
    del_c = obj1.col - obj2.col
    del_r = obj1.row - obj2.row
    if (del_c, del_r) in DICT_LOC_DELTA2NAME:
        return DICT_LOC_DELTA2NAME[(del_c, del_r)]
    return None


class Object:
    def __init__(self, color, shape, row, col):
        self.color = color
        self.shape = shape
        self.row = row
        self.col = col
        self.identify_status = 'color-shape-location'
        self.id_ = self.get_id()

    def __str__(self):
        return 'color: {}; shape: {}; row: {}; col: {}'.format(self.color,
                self.shape, self.row, self.col)

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
        self.d_id_relloc = defaultdict(dict)
        self.d_grid_obj = {}  # currently not maintain all history
        self.grid_size = grid_size

    def size(self):
        return len(self.d_id_obj)

    def select_obj_ref_grid(self, select_empty=True, is_viable=True, error_mode=None):
        if is_viable and self.size() == 0:
            return None, None, None, None
        options = []
        for (row, col), obj_ref in self.d_grid_obj.items():
            for (row_d, col_d), loc_rel in DICT_LOC_DELTA2NAME.items():
                row_adj, col_adj = row - row_d, col - col_d
                if not (0 <= row_adj < self.grid_size and 0 <= col_adj < self.grid_size):
                    continue
                if select_empty and is_viable:
                    if (row_adj, col_adj) not in self.d_grid_obj:
                        options.append((obj_ref, row_adj, col_adj, loc_rel))
                elif not select_empty and is_viable:
                    if (row_adj, col_adj) in self.d_grid_obj:
                        options.append((obj_ref, row_adj, col_adj, loc_rel))
        if not is_viable:
            def f(x, y):
                return random.choice(list(set(y) - set([x])))
            # loc_abs, row, col = self.select_loc_abs(select_empty, is_viable)
            # if loc_abs:
            #     color = random.choice(COLORS)
            #     shape = random.choice(SHAPES)
            #     options.append(Object(color, shape, row, col))
            if self.size() > 0: # wrong reference
                (row, col), obj = random.choice(list(self.d_grid_obj.items()))
                random_selector = random.uniform(0, 1)
                if random_selector >= 0.5:
                    color = f(obj.color, COLORS)
                    shape = random.choice(SHAPES)
                else:
                    color = random.choice(COLORS)
                    shape = f(obj.shape, SHAPES)
                obj_ref = Object(color, shape, obj.row, obj.col)
                for (row_d, col_d), loc_rel in DICT_LOC_DELTA2NAME.items():
                    row_adj, col_adj = row - row_d, col - col_d
                    options.append((obj_ref, row_adj, col_adj, loc_rel))
        if len(options) > 0:
            return random.choice(options)
        return None, None, None, None

    def select_adj_grid(self, row, col, is_viable=None):
        lst_grid = []
        for (row_d, col_d), loc_rel in DICT_LOC_DELTA2NAME.items():
            row_adj = row - row_d
            col_adj = col - col_d
            cond_1 = (0 <= row_adj < self.grid_size and 0 <= col_adj < self.grid_size)
            cond_2 = (row_adj, col_adj) not in self.d_grid_obj
            if is_viable == True and cond_1 and cond_2:
                lst_grid.append((row_adj, col_adj, loc_rel))
            if is_viable == False and not cond_1 and not cond_2:
                lst_grid.append((row_adj, col_adj, loc_rel))
            if is_viable is None:
                lst_grid.append((row_adj, col_adj, loc_rel))
        if len(lst_grid) > 0:
            return random.choice(lst_grid)
        return None, None, None

    def select_loc_abs(self, select_empty=True, is_viable=None):
        l_all = list(DICT_LOC_ABS2NAME.keys())
        l_not_empty = [e for e in self.d_grid_obj.keys() if e]
        l_empty_abs = list(set(l_all) - set(l_not_empty))
        l_not_empty_abs = list(set(l_not_empty).intersection(set(l_all)))
        d_empty_viable = {(True, True): l_empty_abs, (False, True): l_not_empty_abs,
                          (True, False): l_not_empty_abs, (False, False): l_empty_abs}
        if is_viable is None:
            l_selected = l_all
        else:
            l_selected = d_empty_viable[(select_empty, is_viable)]
        if not l_selected:
            return None, None, None
        (row, col) = random.choice(l_selected)
        return DICT_LOC_ABS2NAME[(row, col)], row, col

    def select_loc_rel(self, select_empty=True, is_viable=None):
        obj_ref, row, col, loc_rel = self.select_obj_ref_grid(select_empty, is_viable)
        # obj_ref = self.select_obj_ref(select_empty, is_viable)
        # if obj_ref:
        #     row, col, loc_rel = self.select_adj_grid(obj_ref.row, obj_ref.col, select_empty, is_viable)
        return obj_ref, row, col, loc_rel

    def get_obj_desc(self, obj,  mode_ref=None, mode_loc=None):
        features = self.get_obj_features(obj, mode_ref)
        if len(features) > 0:
            color, shape, loc_abs, loc_rel, obj_ref = random.choice(features)
            tmpl = random.choice(DICT_TEMPLATES[OBJ])
            s = Template(tmpl)
            text = s.substitute(color=xs(color), shape=xs(shape))
            text += ' ' + self.get_loc_desc(loc_abs, loc_rel, obj_ref)
            return re.sub(' +', ' ', text)
        return self.get_obj_ref_desc(obj)

    def get_obj_ref_desc(self, obj):
        features = self.get_obj_features(obj)
        random.shuffle(features)
        for feature in features:
            if feature[3] is None:
                return self.get_obj_desc(obj)
        tmpl = random.choice(DICT_TEMPLATES[OBJ_REF])
        s = Template(tmpl)
        for (row, col), loc_abs in DICT_LOC_ABS2NAME.items():
            row_d = obj.row - row
            col_d = obj.col - col
            if (row_d, col_d) in DICT_LOC_DELTA2NAME:
                loc_rel = DICT_LOC_DELTA2NAME[(row_d, col_d)]
                text = s.substitute(color=obj.color, shape=obj.shape, loc_rel=loc_rel, loc_abs=loc_abs)
                return text
        return ''

    def get_loc_desc(self, loc_abs, loc_rel, obj_ref):
        if loc_abs and loc_rel and obj_ref:
            mode_loc = random.choice(MODES_LOC)
        elif loc_abs:
            mode_loc = LOC_ABS
        elif loc_rel and obj_ref:
            mode_loc = LOC_REL
        else:
            return ''
        tmpl = random.choice(DICT_TEMPLATES[mode_loc])
        s = Template(tmpl)
        text = ''
        if mode_loc == LOC_ABS:
            text = s.substitute(loc_abs=loc_abs)
        if mode_loc == LOC_REL:
            t_obj_ref = self.get_obj_ref_desc(obj_ref)
            text = s.substitute(loc_rel=loc_rel, obj_ref=t_obj_ref)
        return text

    # def get_ref_obj(self):
    #     # randomly select an object identified by color-shape or abs location as reference
    #     available = [obj for obj in self.objects if
    #                  obj.identify_status == 'color-shape' or obj.identify_status == 'location']
    #     assert len(available) > 0
    #     return random.choice(available)

    # def exec_instruction(self, inst):
    #     assert inst.viable(self)
    #     if isinstance(inst, AddInstruction):
    #         self.objects.append(inst.obj)
    #     elif isinstance(inst, PatternAddInstruction):
    #         for obj in inst.objs:
    #             self.objects.append(obj)
    #     elif isinstance(inst, RemoveInstruction):
    #         self.objects.remove(inst.target_obj)
    #     elif isinstance(inst, MoveInstruction):
    #         self.objects.remove(inst.remove_inst.target_obj)
    #         self.objects.append(inst.add_inst.obj)
    #     self.update_obj_identify_status()

    # def update_obj_identify_status(self):
    #     self.color_shape_cnt = Counter(['{}-{}'.format(obj.color, obj.shape) for obj in self.objects])
    #     for obj in self.objects:
    #         if self.color_shape_cnt['{}-{}'.format(obj.color, obj.shape)] == 1:
    #             obj.identify_status = "color-shape"
    #         elif (obj.row, obj.col) in abs_loc_dict.values():
    #             obj.identify_status = "location"
    #         else:
    #             obj.identify_status = "color-shape-location"

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
        if (obj.row, obj.col) in self.d_grid_obj:
            return self.d_grid_obj[(obj.row, obj.col)]
        return STATE_EMPTY

    def update_spatial_ref_add(self, obj_new):
        for id_, obj in self.d_id_obj.items():
            rel = get_rel_loc(obj, obj_new)
            if rel:
                self.d_id_relloc[obj.id_][obj_new.id_] = rel
            rel = get_rel_loc(obj_new, obj)
            if rel:
                self.d_id_relloc[obj_new.id_][obj.id_] = rel

    def update_spatial_ref_delete(self, obj_delete):
        if obj_delete.id_ in self.d_id_relloc:
            del self.d_id_relloc[obj_delete.id_]
        l_del = [k for k, v in self.d_id_relloc.items() if obj_delete.id_ in v]
        for ele in l_del:
            del self.d_id_relloc[ele][obj_delete.id_]

    def update_ref_obj_features(self, obj_rm):  # already removed
        if (obj_rm.color, obj_rm.shape) in self.d_feature_id:
            if len(self.d_feature_id) == 1:
                id_ = self.d_feature_id[(obj_rm.color, obj_rm.shape)]
                features = self.get_obj_features(self.d_id_obj[id_])
                self.d_id_ref_feature[id_] = features

    def is_action_viable(self, obj, action):
        if not (0 <= obj.row < self.grid_size and 0 <= obj.col < self.grid_size):
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
        if self.is_action_viable(obj, ADD):
            self.d_id_obj[obj.id_] = obj
            self.d_grid_obj[(obj.row, obj.col)] = obj  # STATE_OCCU
            self.d_feature_id[(obj.color, obj.shape)].append(obj.id_)
            self.d_feature_id[obj.color].append(obj.id_)
            self.d_feature_id[obj.shape].append(obj.id_)
            # features = self.get_features_obj_ref(obj)
            # self.d_id_ref_feature[obj.id_] = features
            self.update_spatial_ref_add(obj)
            return STATUS_SUCCESSFUL
        else:
            return STATUS_FAILED

    def delete(self, obj):
        if self.is_action_viable(obj, DELETE):
            del self.d_id_obj[obj.id_]
            if obj.id_ in self.d_id_relloc:
                del self.d_id_relloc[obj.id_]
            self.d_feature_id[(obj.color, obj.shape)].remove(obj.id_)
            self.d_feature_id[obj.color].remove(obj.id_)
            self.d_feature_id[obj.shape].remove(obj.id_)
            self.update_ref_obj_features(obj)
            self.update_spatial_ref_delete(obj)
            del self.d_grid_obj[(obj.row, obj.col)]
            return STATUS_SUCCESSFUL
        return STATUS_FAILED

    def move(self, obj_from, obj_to):
        if self.is_action_valid(obj_from, DELETE) and \
           self.is_action_valid(obj_to, ADD):
            return self.delete(obj_from) and self.add(obj_to)
        return STATUS_FAILED

    def get_obj_features(self, obj, mode_ref=None):
        lst = []
        loc_abs = loc_rel = obj_ref = None
        if (obj.row, obj.col) in DICT_LOC_ABS2NAME:
            loc_abs = DICT_LOC_ABS2NAME[(obj.row, obj.col)]
            lst.append((None, None, loc_abs, None, None))
        if obj.id_ in self.d_id_relloc and len(self.d_id_relloc[obj.id_]) > 1:
            obj_ref_id, loc_rel = random.choice(list(self.d_id_relloc[obj.id_].items()))
            if obj_ref_id and obj_ref_id in self.d_id_obj:
                obj_ref = self.d_id_obj[obj_ref_id]
                if (obj_ref.row, obj_ref.col) in DICT_LOC_ABS2NAME:
                    lst.append((None, None, None, loc_rel, obj_ref))
            else:
                loc_rel, obj_ref = None, None
        else:
            for id_, obj_r in self.d_id_obj.items():
                rel = get_rel_loc(obj_r, obj)
                if rel:
                    loc_rel, obj_ref = rel, obj_r
                    break
                rel = get_rel_loc(obj, obj_r)
                if rel:
                    loc_rel, obj_ref = rel, obj_r
                    break
        if mode_ref == MODE_FULL:
            lst_full = []
            if loc_abs:
                lst_full.append((obj.color, obj.shape, loc_abs, None, None))
            if loc_rel and obj_ref:
                lst_full.append((obj.color, obj.shape, None, loc_rel, obj_ref))
            return lst_full
        if obj.color in self.d_feature_id and len(self.d_feature_id[obj.color]) == 1:
            lst.append((obj.color, None, None, None, None))
        if obj.shape in self.d_feature_id and len(self.d_feature_id[obj.shape]) == 1:
            lst.append((None, obj.shape, None, None, None))
        if (obj.color, obj.shape) in self.d_feature_id and len(self.d_feature_id[(obj.color, obj.shape)]) == 1:
            lst.append((obj.color, obj.shape, None, None, None))
        if self.size() == 1:
            lst.append((None, None, None, None, None))
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
                pass
                # data.append({"prev_canvas": prev_canvas,
                #              "current_instruction": current_instuction,
                #              "next_object": next_obj})

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
