import regex as re
import random
from collections import defaultdict


OPS = ['→', '↔', '¬', '⊕', '∨', '∧', '∀', '∃']
BASE = [i.strip() for i in open('dataset/syllo/syllo.txt', encoding='utf-8').readlines()]
CHAIN = defaultdict(set)

for logi_type in ['a', 'e', 'i', 'o']:
    for idx, syllo in enumerate(BASE):
        syllo = syllo.split(',')[:2]
        for s in syllo:
            if s[1] == logi_type:
                CHAIN[logi_type].add(idx)
                break
CHAIN = {k: list(v) for k, v in CHAIN.items()}

def convert_to_fol(syllo_var):

    syllo_type = syllo_var[syllo_var.find(']')+1]
    left, right = syllo_var.split(syllo_type)
    if syllo_type == 'a':
        return f'∀x ({left}(x) → {right}(x))'
    elif syllo_type == 'e':
        return f'∀x ({left}(x) → ¬{right}(x))'
    elif syllo_type == 'i':
        return f'∃x ({left}(x) ∧ {right}(x))'
    elif syllo_type == 'o':
        return f'∃x ({left}(x) ∧ ¬{right}(x))'


def resolve_syllo(syllo, counter, last_syllo):
    tmp_map = {}
    new_var = []
    for symbol, p in zip(syllo[0].split(syllo[0][1]), last_syllo.split(syllo[0][1])):
        tmp_map[symbol] = p
    for symbol in ['S', 'M', 'P']:
        if symbol not in tmp_map:
            tmp_map[symbol] = f'[R{counter}]'
            new_var.append(f'[R{counter}]')
            counter += 1
    result = []
    for statement in syllo:
        for k,v in tmp_map.items():
            statement = statement.replace(k, v)
        result.append(statement)
    return result, counter, new_var

def negation_fol(syllo_var):

    syllo_type = syllo_var[syllo_var.find(']')+1]
    left, right = syllo_var.split(syllo_type)  

    if syllo_type == 'a':
        return f'∃x ({left}(x) ∧ ¬{right}(x))'
    elif syllo_type == 'e':
        return f'∃x ({left}(x) ∧ {right}(x))'
    elif syllo_type == 'i':
        return f'∀x ({left}(x) → ¬{right}(x))'
    elif syllo_type == 'o':
        return f'∀x ({left}(x) → {right}(x))'

def find_next(syllo):
    st = syllo[1]
    next_syllo = BASE[random.choice(CHAIN[st])]
    next_syllo = next_syllo.split(',')
    if st not in next_syllo[0]:
        next_syllo[0], next_syllo[1] = next_syllo[1], next_syllo[0]
    assert st in next_syllo[0], 'Invalid syllogism: {}'.format(next_syllo)
    return next_syllo

def get_syllo(depth=1, entailment=True):
    '''
    Total depth = 3+depth-1
    '''
    assert depth >= 1, 'Depth must be at least 1'
    syllo = random.choice(BASE)
    syllo = syllo.split(',')
    syllo_var = []
    all_vars = {'[R1]', '[R2]', '[R0]'}
    pcounter = 3
    tmp_map = {'S': '[R0]', 'M': '[R1]', 'P': '[R2]'}
    for statement in syllo:
        for k,v in tmp_map.items():
            statement = statement.replace(k, v)
        syllo_var.append(statement)

        
    for i in range(depth-1):
        starter = syllo.pop(-1)
        next_syllo = find_next(starter)
        next_syllo_var, pcounter, new_var = resolve_syllo(next_syllo, pcounter, syllo_var.pop(-1))
        syllo_var.extend(next_syllo_var[1:])
        syllo.extend(next_syllo[1:])
        all_vars.update(new_var)

    
    result = []
    for i in syllo_var:
        result.append(convert_to_fol(i))

    if not entailment:
        result[-1] = negation_fol(syllo_var[-1])
    

    return tuple(result), all_vars


def resolve_predicate(all_vars):
    options = ['⊕', '∨', '∧', '']
    result = {k: random.choice(options) for k in all_vars}
    return result