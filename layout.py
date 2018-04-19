import random
from collections import Counter, defaultdict, OrderedDict
from utils import get_hashcode
from string import Template
import re
from const import *


def xs(o):
    if o is None:
        return ''
    return str(o)


def tmpl2txt_act(act, t_obj, t_loc_abs="", t_loc_rel=""):
    tmpl = random.choice(DICT_TEMPLATES[act])
    s = Template(tmpl)
    text = s.substitute(obj=t_obj, loc_abs=t_loc_abs, loc_rel=t_loc_rel)
    return re.sub(' +', ' ', text)  # remove duplicated spaces


def get_rel_loc(obj1, obj2):
    del_c = obj1.col - obj2.col
    del_r = obj1.row - obj2.row
    if (del_c, del_r) in DICT_LOC_DELTA2NAME:
        del_r = obj1.row - obj2.row
        return DICT_LOC_DELTA2NAME[(del_c, del_r)]
    return None


class Object:
    def __init__(self, color, shape, row, col):
        self.color = color
        self.shape = shape
        self.row = row
        self.col = col
        self.features = None
        self.id_ = self.get_id()

    def __str__(self):
        return 'color: {}; shape: {}; row: {}; col: {}'.format(self.color,
                self.shape, self.row, self.col)

    def get_id(self):
        id_ = get_hashcode([self.color, self.shape, self.row, self.col])
        return id_

    def to_dict(self):
        d = {'color': self.color, 'shape': self.shape, 'row': self.row, 'col': self.col}
        return d

    def get_info(self):
        return {'row': self.row, 'col': self.col, 'color': self.color, 'shape': self.shape}


class Canvas:
    def __init__(self, grid_size=GRID_SIZE):
        self.d_id_obj = {}
        self.d_feature_id = defaultdict(list)  # key (color, shape)
        self.d_id_ref_feature = defaultdict(list)
        self.d_id_rel = defaultdict(dict)
        self.d_grid_obj = {}  # currently not maintain all history
        self.grid_size = grid_size

    def size(self):
        return len(self.d_id_obj)

    def select_obj_ref_grid(self, select_empty=True, is_viable=True, excluded_grids=[]):
        if is_viable and self.size() - len(excluded_grids) <= 0:
            return None, None, None, None
        options = []
        for (row, col), obj_ref in self.d_grid_obj.items():
            if (row, col) in excluded_grids:
                continue
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
            if len(self.d_grid_obj) > 0: # wrong reference
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

    def get_obj_desc(self, obj,  mode_ref=None):
        features = self.get_obj_features(obj, mode_ref)
        if len(features) > 0:
            color, shape, loc_abs, loc_rel, obj_ref = random.choice(features)
            obj.features = {'color': obj.color, 'shape': obj.shape,
                            'loc_abs': loc_abs, 'loc_rel': loc_rel,
                            'obj_ref': obj_ref.to_dict() if obj_ref else None}
            tmpl = random.choice(DICT_TEMPLATES[OBJ])
            s = Template(tmpl)
            text = s.substitute(color=xs(color), shape=xs(shape))
            text += ' ' + self.get_loc_desc(loc_abs, loc_rel, obj_ref, mode_ref)
            return re.sub(' +', ' ', text)
        return self.get_obj_ref_desc(obj)


    def get_obj_ref_desc(self, obj, mode_ref=None):
        # features = self.get_obj_features(obj)
        # random.shuffle(features)
        # for feature in features:
        #     if feature[3] is None:
        #         return self.get_obj_desc(obj, mode_ref)
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

    def get_loc_desc(self, loc_abs, loc_rel, obj_ref, mode_ref=None, mode_loc=None):
        options = []
        if loc_abs:
            options.append(LOC_ABS)
        if loc_rel and obj_ref:
            options.append(LOC_REL)
        if mode_loc not in options and len(options) > 0:
            mode_loc = random.choice(options)
        if mode_loc is None:
            return ''
        tmpl = random.choice(DICT_TEMPLATES[mode_loc])
        s = Template(tmpl)
        text = ''
        if mode_loc == LOC_ABS:
            text = s.substitute(loc_abs=loc_abs)
        if mode_loc == LOC_REL:
            t_obj_ref = self.get_obj_ref_desc(obj_ref, mode_ref)
            text = s.substitute(loc_rel=loc_rel, obj_ref=t_obj_ref)
        return text

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
                self.d_id_rel[obj.id_][obj_new.id_] = rel
            rel = get_rel_loc(obj_new, obj)
            if rel:
                self.d_id_rel[obj_new.id_][obj.id_] = rel

    def update_spatial_ref_delete(self, obj_delete):
        if obj_delete.id_ in self.d_id_rel:
            del self.d_id_rel[obj_delete.id_]
        l_del = [k for k, v in self.d_id_rel.items() if obj_delete.id_ in v]
        for ele in l_del:
            del self.d_id_rel[ele][obj_delete.id_]

    def update_ref_obj_features(self, obj_rm):  # already removed
        if (obj_rm.color, obj_rm.shape) in self.d_feature_id:
            if len(self.d_feature_id) == 1:
                id_ = self.d_feature_id[(obj_rm.color, obj_rm.shape)]
                features = self.get_obj_features(self.d_id_obj[id_])
                self.d_id_ref_feature[id_] = features

    def is_action_viable(self, obj, act):
        if not (0 <= obj.row < self.grid_size and 0 <= obj.col < self.grid_size):
            return False
        if act == ADD:
            state = self.get_state_by_coordinates(obj)
            if state == STATE_OCCU:
                return False
            return True
        if act == DELETE:
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
            self.update_spatial_ref_add(obj)
            return STATUS_SUCCESSFUL
        else:
            return STATUS_FAILED

    def delete(self, obj):
        if self.is_action_viable(obj, DELETE):
            if obj.id_ in self.d_id_obj:
                del self.d_id_obj[obj.id_]
            if (obj.row, obj.col) in self.d_grid_obj:
                del self.d_grid_obj[(obj.row, obj.col)]
            if obj.id_ in self.d_id_rel:
                del self.d_id_rel[obj.id_]
            self.d_feature_id[(obj.color, obj.shape)].remove(obj.id_)
            self.d_feature_id[obj.color].remove(obj.id_)
            self.d_feature_id[obj.shape].remove(obj.id_)
            self.update_ref_obj_features(obj)
            self.update_spatial_ref_delete(obj)
            return STATUS_SUCCESSFUL
        return STATUS_FAILED

    # def move(self, obj_from, obj_to):
    #     if self.is_action_valid(obj_from, DELETE) and \
    #        self.is_action_valid(obj_to, ADD):
    #         return self.delete(obj_from) and self.add(obj_to)
    #     return STATUS_FAILED

    def get_obj_features(self, obj, mode_ref=None):
        # feature: color, shape, loc_abs, loc_rel, obj_ref
        lst = []
        loc_abs = loc_rel = obj_ref = None
        if (obj.row, obj.col) in DICT_LOC_ABS2NAME:
            loc_abs = DICT_LOC_ABS2NAME[(obj.row, obj.col)]
            lst.append((None, None, loc_abs, None, None))
        if obj.id_ in self.d_id_rel and len(self.d_id_rel[obj.id_]) > 1:
            obj_ref_id, loc_rel = random.choice(list(self.d_id_rel[obj.id_].items()))
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