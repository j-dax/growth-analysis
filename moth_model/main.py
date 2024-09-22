from dataclasses import dataclass
from typing import Callable
from sympy import sympify, pprint, Symbol
from sympy.abc import v, w, x, y, z
from numpy import prod, array, append, format_float_positional, vectorize
from random import randint
from pandas import DataFrame


def describe_bins(f_base: sympify, iterations: int, *generators):
    arr = []

    for it in range(iterations):
        f = f_base.copy()
        args = [(gen.sym, gen.gen()) for gen in generators]
        f = f.subs(args)

        # only one number here after substitution and simplifying
        for i, atom in enumerate(f.atoms()):
            num = atom
            if i > 0:
                print("Unexpected number of atoms found\n\t")
                pprint(f)
                exit(1)
        arr.append(num)

    @vectorize
    def no_sigfigs(x):
        s = format_float_positional(x, 0)
        if s[-1] == '.': s = s[:-1]
        return int(s)
    
    arr = no_sigfigs(array(sorted(arr)))
    return arr

@dataclass
class SymbolGenerator:
    gen: Callable[[], int]
    sym: Symbol

def analyze(is_follower: bool, is_mod_or_vip: bool, sub_tier: int):
    # Scoring methods
    base_selector = lambda : randint(10, 10000)
    follow_producer = lambda : randint(2, 10) if is_follower else 1
    vip_producer = lambda : randint(2, 10) if is_mod_or_vip else 1
    sub_producer = lambda : prod([\
        randint(2, 10) for _ in range(sub_tier)\
    ])
    credit_producer = lambda : 2

    # Containers, used in blind generation
    base = SymbolGenerator(base_selector, v)
    follow = SymbolGenerator(follow_producer, v)
    vip = SymbolGenerator(vip_producer, x)
    sub = SymbolGenerator(sub_producer, y)
    credit = SymbolGenerator(credit_producer, z)

    # The function we'll use
    f = base.sym * follow.sym * sub.sym * vip.sym * credit.sym
    return describe_bins(f, 5000, base, follow, sub, vip, credit)

def human_readable(x):
    suffix = ['', 'k', 'M', 'G', 'T', 'E', 'Z', 'Y', 'R', 'Q']
    suffix_index = 0
    while x > 1000:
        x /= 1000
        suffix_index += 1
    return f"{x:.3f}{suffix[suffix_index]}"

if __name__ == "__main__":
    for is_mod_or_vip in [False, True]:
        for is_follower in [False, True]:
            # store the info by follower/vip status
            data = []
            for sub_tier in range(4):
                data.append(analyze(is_follower, is_mod_or_vip, sub_tier))
            # transpose this to fit on the sub tier axis: 0, 1, 2, 3
            df = DataFrame(array(data).T, columns=['Nonsub', 'T1', 'T2', 'T3'])
            df = df.describe(percentiles=[.25, .5, .75])
            df = df.map(lambda s : human_readable(s))
            # the header
            follower_str = "Follower"
            if not is_follower: follower_str = "Non" + follower_str
            vip_str = "VIP"
            if not is_mod_or_vip: vip_str = "Non" + vip_str
            header = "/".join([follower_str, vip_str])
            print(header)
            print(df)
            print()
