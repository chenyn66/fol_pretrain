import regex as re
import random
import json
from collections import defaultdict


OPS = ['→', '↔', '¬', '⊕', '∨', '∧', '∀', '∃']
BASE = [i.strip() for i in open('dataset/syllo/syllo.txt', encoding='utf-8').readlines()]
CHAIN = defaultdict(set)
IDX2LETTER = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
GREEK=[chr(code) for code in range(945,970)]
NOUNS = set()
ADJS = set()

with open('dataset/syllo/nouns.txt') as f:
    for line in f:
        line = line.strip()
        if line:
            NOUNS.add(line)

NOUNS = list(NOUNS)

with open('dataset/syllo/adjs.txt') as f:
    for line in f:
        line = line.strip()
        if line:
            ADJS.add(line.lower())

ADJS = list(ADJS)

for logi_type in ['a', 'e', 'i', 'o']:
    for idx, syllo in enumerate(BASE):
        syllo = syllo.split(',')[:2]
        for s in syllo:
            if s[1] == logi_type:
                CHAIN[logi_type].add(idx)
                break
CHAIN = {k: list(v) for k, v in CHAIN.items()}


TEMPLATE = {
    'a': [],
    'e': [],
    'i': [],
    'o': []
}

with open('dataset/syllo/template.txt') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if ':' in line:
            st = line[0]
        else:
            assert '{LEFT}' in line and '{RIGHT}' in line, f'{line} has wrong format'
            TEMPLATE[st].append(line.strip())

ADJ_TEMPLATE = {
    'a': [],
    'e': [],
    'i': [],
    'o': []
}

with open('dataset/syllo/adj_template.txt') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if ':' in line:
            st = line[0]
        else:
            assert '{LEFT}' in line and '{RIGHT}' in line, f'{line} has wrong format'
            ADJ_TEMPLATE[st].append(line.strip())

            if st == 'e' :
                for t in ADJ_TEMPLATE['a']:
                    if 'not ' not in t:
                        new_t = t.replace(' {RIGHT}', ' not {RIGHT}')
                        if new_t not in ADJ_TEMPLATE['e']:
                            ADJ_TEMPLATE['e'].append(new_t)
            if st == 'o' :
                for t in ADJ_TEMPLATE['i']:
                    if 'not ' not in t:
                        new_t = t.replace(' {RIGHT}', ' not {RIGHT}')
                        if new_t not in ADJ_TEMPLATE['o']:
                            ADJ_TEMPLATE['o'].append(new_t)



def convert_to_fol(syllo_var, variable):

    syllo_type = syllo_var[syllo_var.find(']')+1]
    left, right = syllo_var.split(syllo_type)
    left, right = variable[left], variable[right]
    left, right = left.replace(' ', '_'), right.replace(' ', '_')
    if syllo_type == 'a':
        return f'∀x ({left}(x) → {right}(x))'
    elif syllo_type == 'e':
        return f'∀x ({left}(x) → ¬{right}(x))'
    elif syllo_type == 'i':
        return f'∃x ({left}(x) ∧ {right}(x))'
    elif syllo_type == 'o':
        return f'∃x ({left}(x) ∧ ¬{right}(x))'

def question2fol(question, variable):
    result = dict()
    for k,v in question.items():
        if k == 'conclusion':
            result[k] = convert_to_fol(v, variable)
        else:
            result[k] = [convert_to_fol(i, variable) for i in v]
    return result

def convert_to_template(syllo_var, variable, rand=False, noun=True):
    # TODO: do we want to use GPT-3 to augment the template? e.g., Paraphrase: xxx
    syllo_type = syllo_var[syllo_var.find(']')+1]
    left, right = syllo_var.split(syllo_type)
    left, right = variable[left], variable[right]
    to_use = TEMPLATE if noun else ADJ_TEMPLATE 
    if rand:
        t = random.choice(to_use[syllo_type])
    else:
        t = to_use[syllo_type][0]
    return t.format(LEFT=left, RIGHT=right).capitalize()



def question2template(question, variable, rand=False, noun=True):
    result = dict()
    for k,v in question.items():
        if k == 'conclusion':
            result[k] = convert_to_template(v, variable, rand, noun=noun)
        else:
            result[k] = [convert_to_template(i, variable, rand, noun=noun) for i in v]
    return result


def random_assign_nouns(variable):
    nouns = [i for i in NOUNS]
    random.shuffle(nouns)
    for k,v in variable.items():
        variable[k] = nouns.pop()
    return variable

def random_assign_adjs(variable):
    adjs = [i for i in ADJS]
    random.shuffle(adjs)
    for k,v in variable.items():
        variable[k] = adjs.pop()
    return variable

def assign_greek_letters(variable):
    return {vn:GREEK[idx] for idx,vn in enumerate(sorted(variable, key=lambda x: int(x[2:-1])))}


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

def negation_syllo(syllo_var):
    syllo_type = syllo_var[syllo_var.find(']')+1]
    left, right = syllo_var.split(syllo_type)  

    if syllo_type == 'a':
        return f'{left}o{right}'
    elif syllo_type == 'e':
        return f'{left}i{right}'
    elif syllo_type == 'i':
        return f'{left}e{right}'
    elif syllo_type == 'o':
        return f'{left}a{right}'

def negate_quesion(question):
    return {
        'story': question['story'],
        'conclusion': negation_syllo(question['conclusion']),
    }

def find_next(syllo):
    st = syllo[1]
    next_syllo = BASE[random.choice(CHAIN[st])]
    next_syllo = next_syllo.split(',')
    if st not in next_syllo[0]:
        next_syllo[0], next_syllo[1] = next_syllo[1], next_syllo[0]
    assert st in next_syllo[0], 'Invalid syllogism: {}'.format(next_syllo)
    return next_syllo

def get_syllo(depth=1):
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

    
    conclusion = syllo_var.pop(-1)
    question = {
        'story': syllo_var,
        'conclusion': conclusion
    }
    all_vars = {vn:IDX2LETTER[idx] for idx,vn in enumerate(sorted(all_vars, key=lambda x: int(x[2:-1])))}
    

    return question, all_vars


def resolve_predicate(all_vars):
    options = ['⊕', '∨', '∧', '']
    result = {k: random.choice(options) for k in all_vars}
    return result