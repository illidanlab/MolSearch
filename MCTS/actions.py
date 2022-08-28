import copy

import bottleneck
import numpy as np

from scores import *

from design_moves import DesignMove
replaceRule = DesignMove("chemblDB3.sqlitdb")


def get_mo_actions(actions, functions, thresholds, t=1.0):
    valid_actions = []
    n_f = len(functions)
    for s in actions:
        mol = Chem.MolFromSmiles(s)
        scores = np.zeros(n_f)
        for i in range(n_f):
            scores[i] = functions[i](mol)
        if np.all(scores >= t * thresholds):
            valid_actions.append(s)
    print("valid actions after constraint {:d}".format(len(valid_actions)))
    return valid_actions


def get_mo_stage2_actions(current_smiles, actions, functions, thresholds, t=1.0):
    ref_size = Chem.MolFromSmiles(current_smiles).GetNumAtoms()
    valid_actions = []
    n_f = len(functions)
    for s in actions:
        mol = Chem.MolFromSmiles(s)
        mol_size = mol.GetNumAtoms()
        scores = np.zeros(n_f)
        for i in range(n_f):
            scores[i] = functions[i](mol)
        
        if np.all(scores >= t * thresholds) and mol_size < ref_size:
            valid_actions.append(s)
    print("valid actions after constraint {:d}".format(len(valid_actions)))
    return valid_actions


def constraint_top_k(valid_actions, score_func, k=10):
    if len(valid_actions) <= k:
        return valid_actions
    scores = []
    for s in valid_actions:
        scores.append(score_func(Chem.MolFromSmiles(s)))
    scores = np.array(scores)
    assert len(scores)==len(valid_actions)

    all_tuple = [(-scores[idx], valid_actions[idx]) for idx in range(len(scores))]
    topk_tuple = sorted(all_tuple)[:k]
    topk_actions = [t[1] for t in topk_tuple]
    return topk_actions


def get_actions(state):
    mol = Chem.MolFromSmiles(state)
    if mol is None:
        raise ValueError("Received invalid state: %s" % state)

    valid_actions = set()
    try:
        print("replace action calculation")
        valid_tmp = _frag_substitution(state, replaceRule)
        print("possible actions: {:d}".format(len(valid_tmp)))
        valid_actions.update(valid_tmp)
    except:
        pass

    return list(valid_actions)


def _frag_substitution(smi, rule, min_pairs=1):
    substitution_actions = rule.one_step_move(query_smi=smi, min_pairs=min_pairs)
    return set(substitution_actions)


if __name__ == '__main__':
    s = 'C=C(C(C)=C(CCCCCCC)CCCCCCCC)C(CC)=C(C)CCCCCCCCC'
    valid_actions = get_valid_actions(s)
    topk_actions = top_k(valid_actions, plogp)
    # scores = []
    # for a in valid_actions:
    #     scores.append(plogp(Chem.MolFromSmiles(a)))
    # print(sorted(scores)[-10:])
    for a in topk_actions:
        print(a, plogp(Chem.MolFromSmiles(a)))

