from enum import Enum, auto


class Obj(Enum):
    MASS = auto()
    FRICTION = auto()
    MOVEABLE = auto()
    SCALE = auto()

    OBSTACLE = auto()
    ORIENT = auto()
    SPATIALRELATE = auto()

    LOCK = auto()
    KINEMATIC = auto()
    SEQ_NAV = auto()

    GEOMETRY = auto()
    MESH = auto()

class Robot(Enum):
    MORPH = auto()
    PERSPECTIVE = auto()
    JOINT_AWARE = auto()

    DYN_TUNE = auto()
    ACT_NAV = auto()
    SKIL_APAPT = auto()


class Inter(Enum):
    TOOL = auto()
    FAIL_ADAPT = auto()
    PLAN = auto()
    HISTORY = auto()
