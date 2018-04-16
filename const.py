DOMAIN_2DSHAPE = '2Dshape'
OTHER_ROLE = {'agent': 'user', 'user': 'agent'}
# painter -- agent / instructor -- user

COLORS = ['red', 'green', 'blue']
SHAPES = ['square', 'circle', 'triangle']

MODE_FULL = 'full'
MODE_MIN = 'min'
MODES_REF = [MODE_MIN, MODE_FULL]

SINGLE = 'single'
MULTI = 'multi'
PATTERN = 'pattern'
MODES_STYLE = [SINGLE, PATTERN]

PATTERN_STYLE = ['row', 'column']

LOC_ABS = 'location_absolute'
LOC_REL = 'location_relative'
MODES_LOC = [LOC_ABS, LOC_REL]


# state of grid
STATE_EMPTY = 0
STATE_OCCU = 1
STATES = [STATE_EMPTY, STATE_OCCU]

STATUS_SUCCESSFUL = True
STATUS_FAILED = False

OBJ = 'object'
OBJ_REF = 'object_reference'
ADD = 'add'
DELETE = 'delete'
MOVE = 'move'
ACTIONS = [ADD, DELETE, MOVE]

GRID_SIZE = 5

DICT_NAME2LOC_ABS = {
    'top-left': (0, 0),
    'top-right': (0, GRID_SIZE - 1),
    'bottom-left': (GRID_SIZE - 1, 0),
    'bottom-right': (GRID_SIZE - 1, GRID_SIZE - 1),
    'center': (GRID_SIZE // 2, GRID_SIZE // 2),
    'top-middle': (0, GRID_SIZE // 2),
    'botton-middle': (GRID_SIZE - 1, GRID_SIZE // 2),
    'left-middle': (GRID_SIZE // 2, 0),
    'right-middle': (GRID_SIZE // 2, GRID_SIZE - 1)
}

DICT_LOC_ABS2NAME = {v: k for k, v in DICT_NAME2LOC_ABS.items()}

DICT_LOC_DELTA2NAME = {
    (1, 0): 'top-of',
    (-1, 0): 'bottom-of',
    (0, -1): 'right-of',
    (0, 1): 'left-of',
    (1, 1): 'top-left-of',
    (1, -1): 'top-right-of',
    (-1, 1): 'bottom-left-of',
    (-1, -1): 'bottom-right-of',
}

TMPL_ADD = ["add one $obj", #  $loc_abs $loc_rel
            "now place a $obj"]

TMPL_DEL = ["now remove the $obj from the canvas", "delete the $obj", "please get rid of the $obj"]

TMPL_MV = ["", ""]

TMPL_LOC_ABS = ["at $loc_abs of the canvas"]

TMPL_LOC_REL = ["at $loc_rel of the $obj_ref"]

TMPL_OBJ_REF = ["$color $shape at $loc_rel of the $loc_abs part of the canvas"]

TMPL_OBJ = ["$color $shape object ", "$color $shape one "]

DICT_TEMPLATES = {ADD: TMPL_ADD,
                  DELETE: TMPL_DEL,
                  MOVE: TMPL_MV,
                  OBJ: TMPL_OBJ,
                  OBJ_REF: TMPL_OBJ_REF,
                  LOC_ABS: TMPL_LOC_ABS,
                  LOC_REL: TMPL_LOC_REL}
