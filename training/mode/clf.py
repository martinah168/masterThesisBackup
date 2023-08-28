from enum import Enum


class ClfMode(str, Enum):
    one_vs_all = 'one_vs_all'
    one_vs_one = 'one_vs_one'
    multi_class = 'multi_class'
