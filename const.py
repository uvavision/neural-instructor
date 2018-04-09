# mode for generating instructions
MODE_FULL = 'full'
MODE_MIN = 'min'
MODES = [MODE_MIN, MODE_FULL]

# state of grid
STATE_EMPTY = 0
STATE_OCCU = 1
STATES = [STATE_EMPTY, STATE_OCCU]

STATUS_SUCCESSFUL = True
STATUS_FAILED = False


ADD = 'add'
DELETE = 'delete'
MOVE = 'move'
ACTIONS = [ADD, DELETE, MOVE]

GRID_SIZE = 5

DICT_ABC_LOC = {
    'top-left': (0, 0),
    'top-right': (0, GRID_SIZE - 1),
    'bottom-left': (GRID_SIZE - 1, 0),
    'bottom-right': (GRID_SIZE - 1, GRID_SIZE - 1),
    'center': (GRID_SIZE // 2, GRID_SIZE // 2)
}

DICT_DEL_REL_LOC = {
    (1, 0): 'top-of',
    (-1, 0): 'bottom-of',
    (0, -1): 'right-of',
    (0, 1): 'left-of',
    (1, 1): 'top-left-of',
    (1, -1): 'top-right-of',
    (-1, 1): 'bottom-left-of',
    (-1, -1): 'bottom-right-of',
}

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
