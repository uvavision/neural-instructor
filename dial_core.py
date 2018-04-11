from const import *
from domain_2D_shape import *
from collections import OrderedDict
from synthetic_layout import Object, Canvas

MODES = [SINGLE, MULTI, PATTERN]


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

    def get_act_obj(self, actions=[ADD], mode=None):
        if not mode:
            mode = random.choice(MODES)
        for action in actions:



    def get_next_message(self, actions_pre, dial_act_pre, role_pre):
        actions = []
        role = OTHER_ROLE[role_pre]
        return role, actions_pre


if __name__ == '__main__':
    pass
