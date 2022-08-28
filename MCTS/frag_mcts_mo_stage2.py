"""
code adapted from https://github.com/jensengroup/GB-GM/blob/master/GB-GM-MCTS.py by Jan H. Jensen 2019
"""

from actions import get_actions, get_mo_actions, get_mo_stage2_actions
from scores import *

import numpy as np
import math

from rdkit import Chem

import random

import hashlib
import argparse
import time, os, datetime
import pickle as pkl

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--goal', type = str, default = 'qed_sa')
parser.add_argument('--constraint', type = str, default = 'gsk3b_jnk3')
parser.add_argument('--start_mols', type = str, default = 'task1')
# parser.add_argument('--mol_idx', type = int, default = 0)
# parser.add_argument('--start_idx', type = int, default = 0)
# parser.add_argument('--end_idx', type = int, default = 1)
parser.add_argument('--group_idx', type = int, default = 0)
parser.add_argument('--max_child', type = int, default = 3)
parser.add_argument('--num_sims', type = int, default = 10)
parser.add_argument('--scalar', type = float, default = 0.7)
parser.add_argument('--seed', type = int, default = 1)
args = parser.parse_args()
print(args.goal)
print(__file__)

def get_score_function(name):
    if name == 'plogp':
        sf = plogp
    elif name == 'qed':
        sf = qed
    elif name == 'esol':
        sf = esol
    elif name == 'sa':
        sf = sa
    elif name == 'rges':
        sf = rges
    elif name == 'gsk3b':
        sf = gsk3b
    elif name == 'jnk3':
        sf = jnk3
    elif name == 'npc1':
        sf = npc1
    elif name == 'insig1':
        sf = insig1
    elif name == 'hmgcs1':
        sf = hmgcs1
    else:
        print("invalid goal!")
    return sf

goals = args.goal.split("_")
constraints = args.constraint.split("_")
print(goals)
print(constraints)
functions = []
for g in goals:
    functions.append(get_score_function(g))

constraint_functions = []
for c in constraints:
    constraint_functions.append(get_score_function(c))

n_obj = len(goals)
MAX_LEVEL = 5

# topk_actions how to pick (rank based on qed because qed cannot be low, its not like pareto front where low value is acceptable)
# reward --> d dimension
# score --> d dimension
# selection --> pareto front, then uniform random
# how to keep track of max_score (reward update)? if non-dominant, add to the list
# max score format [[smiles, [r1, r2, ..]]]

class Node():

    def __init__(self, state, n_obj=n_obj, max_child=args.max_child, parent=None):
        self.visits = 1
        self.reward = np.zeros(n_obj) # float?
        self.state = state
        self.parent = parent
        self.max_child = max_child
        self.children = []

    def add_child(self, child_state):
        child = Node(child_state, parent=self)
        self.children.append(child)

    def update(self, r):
        # r is a vector
        self.reward += r
        self.visits += 1

    def fully_expanded(self):
        if len(self.children) == self.max_child:
            return True
        return False

    def __repr__(self):
        # s = "visits {:d} reward {:.2f} num_child {:d} state {:s}".format(self.visits, self.reward, len(self.children), self.state.smiles)
        # s = str(self.state.smiles)
        s = "{:s} {:d} ".format(self.state.smiles, self.visits)
        tmp = "[ "
        for i in range(self.reward.shape[0]):
            tmp += str(self.reward[i])
            tmp += " "
        tmp += "]"
        s += tmp
        return s


class State():

    def __init__(self, smiles='', level=0):
        self.smiles = smiles
        self.level = level
        self.score = []
        if self.smiles:
            print("smiles score ")
            mol = Chem.MolFromSmiles(self.smiles)
            for f in functions:
                self.score.append(f(mol))
        else:
            print("smiels score else")
            self.score = [0.0] * n_obj

        self.valid_actions = self.get_valid_actions()

    def get_valid_actions(self):
        actions = get_actions(state=self.smiles, allow_substitution=True, allow_atom_addition=False, allow_bond_addition=False)
        mo_actions = get_mo_stage2_actions(self.smiles, actions, functions=functions, thresholds=np.ones(n_obj)*0.5, t=1.0)   #change 1
        return mo_actions #constraint_top_k(valid_actions_constraint, score_func=target_function, k=args.top_k)

    def next_state(self):
        if len(self.valid_actions) == 0:
            self.level = MAX_LEVEL
            return self
        #print(self.valid_actions)
        s = random.choice(self.valid_actions)
        next = State(s, self.level + 1)
        return next

    def terminal(self):
        return self.level == MAX_LEVEL

    def dominate_score(self, v):
        d = len(self.score)
        r = [1.0] * d
        for i in range(d):
            if self.score[i] < v[i]:
                r[i] = 0.0
        return np.array(r)

    def reward(self):
        global max_score
        global count
        global val_score
        count += 1

        #if constraint_functions[0](Chem.MolFromSmiles(self.smiles))>=0.5 and constraint_functions[1](Chem.MolFromSmiles(self.smiles)) >=0.5:
        #if (self.score[0] >= 0.6) and (self.score[1] >= 0.67): ##change 2
        if np.all(np.array(self.score) >= 0.5):
            if self.smiles not in val_score:
                val_score[self.smiles] = self.score

        # max_score dictionary: key smiles, value [r1, r2,..]
        total_reward = np.zeros(len(self.score))
        n = len(max_score)
        keys_to_delete = []
        new_max_score_found = False
        always_uncomparable = True
        for s, r in max_score.items():
            win = self.dominate_score(r)
            total_reward += win
            if sum(win) == float(len(self.score)):
                keys_to_delete.append(s)
                new_max_score_found = True
                always_uncomparable = False
            elif sum(win) == 0.0:
                always_uncomparable = False

        for k in keys_to_delete:
            max_score.pop(k, None)
        if new_max_score_found or always_uncomparable:
            if self.smiles not in max_score:
                max_score[self.smiles] = self.score
        return total_reward/n


    def __hash__(self):
        # override default built-in function to compare whether two objects are the same
        return int(hashlib.md5(str(self.smiles).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        s = "level {:d} state {:s} score ".format(self.level, self.smiles)
        tmp = "[ "
        for r in self.score:
            tmp += str(r)
            tmp += " "
        tmp += "]"
        s += tmp
        return s


def SIMULATION(state):
    # default poclicy, simulation phase
    # return final reward
    while state.terminal() == False:
        #print(state.smiles, plogp(Chem.MolFromSmiles(state.smiles)), Chem.MolFromSmiles(state.smiles).GetNumAtoms())
        state = state.next_state()
        ## examine here

    return state.reward()   # not node


def BACKPROP(node,reward):
    # backpropgation phase
    # return nothing
    while node != None:
        node.update(reward)
        node = node.parent
    return


def dominate(v1, v2):
    return (v1 > v2).all()


def BESTCHILD(node, scalar=0.3): #1/math.sqrt(2.0)
    # place 1: tree policy, selection?
    # place 2: uct search, return best child so far
    # mobj need to modify
    best_d = dict()
    for i in range(len(node.children)):
        c = node.children[i]
        exploit = c.reward / c.visits
        explore = math.sqrt(2.0 * math.log(node.visits) + 0.5 * math.log(len(node.reward)) / float(c.visits))
        score = exploit + scalar * explore #vector
        best_d[c] = score

    first_run = True
    last_len = len(best_d)
    while len(best_d) != last_len or first_run:
        first_run = False
        last_len = len(best_d)

        keys = list(best_d.keys())
        keys_to_delete = []
        for i in range(len(keys)-1):
            c1 = keys[i]
            s1 = best_d[c1]
            for j in range(i, len(keys)):
                c2 = keys[j]
                s2 = best_d[c2]

                if dominate(s1, s2):
                    keys_to_delete.append(c2)
                elif dominate(s2, s1):
                    keys_to_delete.append(c1)
                else:
                    pass
        for k in keys_to_delete:
            best_d.pop(k, None)

    best_children = list(best_d.keys())
    if len(best_children) == 0:
        print("OOPS: no best child found, probably fatal")
    return random.choice(best_children)


def EXPAND(node):
    # expand one time, if not exist, add to child list
    tried_children = [c.state for c in node.children]
    new_state = node.state.next_state()
    lcount = 0
    while new_state in tried_children and lcount < node.max_child: #new_state.max_children
        lcount += 1
        new_state = node.state.next_state()
    node.add_child(new_state)
    return node


def EXPAND_ALL(node):
    # expand until horizontal max is reached
    lcount = 0
    while not node.fully_expanded() and lcount < node.max_child:
        lcount += 1
        node = EXPAND(node)
    #print([n.state.smiles for n in node.children])
    return node


def TREEPOLICY(node):
    # input: root node
    # output: front node
    # selection + expansion

    while node.fully_expanded():
        print("node {:s} {:d} has fully expanded, next...".format(node.state.smiles, node.visits))
        node = BESTCHILD(node, args.scalar)  # may try different scalar at different level

    print("node {:s} not fully expanded, expand and return".format(node.state.smiles))
    if node.state.terminal():
        print("node {:s} terminal, cannot expand, return".format(node.state.smiles))
        return node
    else:
        node = EXPAND_ALL(node)
        print("node {:s} just expanded and return with children: ".format(node.state.smiles))
        print(node.children)
        return node


def UCTSEARCH(budget, root):
    # n_budget simulations
    # for each simulation, reach a node, do sth, backprop one time
    # at last, search for best child and return

    for iter in range(budget):
        print("===== sim_iter {:d} ======".format(iter))
        front = TREEPOLICY(root)  # best child + expand, return front node

        print("front node: ", front.state.smiles, front.reward, front.visits)
        print("children: ", front.children)

        for child in front.children:
            print("simulation on child {:s} ".format(child.state.smiles))
            reward = SIMULATION(child.state)
            print("get reward ", reward)
            BACKPROP(child, reward)
            print("back propagate..")

        for k, v in max_score.items():
            print("max score: ", v, k, Chem.MolFromSmiles(k).GetNumAtoms())
        for k, v in val_score.items():
            print("val score: ", v, k, Chem.MolFromSmiles(k).GetNumAtoms())
        
    return root # BESTCHILD(root, 0) # exploration_scalar=0

start_mols_fn = 'libs/'+args.goal+'_stage1/result_start_mols_' + args.start_mols + '_seed_' + str(args.seed) + '.csv'
start_mols_df = pd.read_csv(start_mols_fn)
start_mol_list = start_mols_df['smiles'].tolist()

save_dir = 'results/' + args.goal + '_stage2/' + args.start_mols + '/seed_' + str(args.seed)
if not os.path.exists(save_dir):
    os.system('mkdir -p {:s}'.format(save_dir))

group_length = 20 #change for different jobs
n_mols = len(start_mol_list)
start_idx = args.group_idx * group_length 
end_idx = min((args.group_idx+1) * group_length, n_mols)

for mol_idx in range(start_idx, end_idx):
    model_str = args.goal + '_maxchild_' + str(args.max_child) + '_sim_' + str(args.num_sims) + '_scalar_' + str(args.scalar) +  '_idx_' + str(mol_idx) + '_seed_' + str(args.seed)

    fn = save_dir + "/" + model_str + ".txt"
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if os.path.exists(fn):
        os.system('mv {:s} {:s}'.format(fn, fn+".bak-{:s}".format(nowTime)))

    with open(fn, "a") as f:
        s = 'size smiles '
        for i in range(n_obj):
            s += goals[i] + ' '
        s += 'qed sa max_or_val \n'
        f.write(s)

    random.seed(args.seed)
    t0 = time.time()

    smiles = start_mol_list[mol_idx]
    print("start molecules: ", smiles)
    root = Node(State(smiles))

    max_score = {smiles: np.array(root.state.score)} #np.ones(n_obj) * (-99.0)
    val_score = dict()
    count = 0

    current_node = UCTSEARCH(args.num_sims, root)

    with open(fn, "a") as f:
        for s, r in max_score.items():
            l = str(Chem.MolFromSmiles(s).GetNumAtoms()) + ' ' + s + ' '
            for x in r:
                l += str(x) + ' '
            l += str(qed(Chem.MolFromSmiles(s))) + ' '
            l += str(sa(Chem.MolFromSmiles(s))) + ' '
            l += str(1)
            l += '\n'
            f.write(l)

        for s, r in val_score.items():
            l = str(Chem.MolFromSmiles(s).GetNumAtoms()) + ' ' + s + ' '
            for x in r:
                l += str(x) + ' '
            l += str(qed(Chem.MolFromSmiles(s))) + ' '
            l += str(sa(Chem.MolFromSmiles(s))) + ' '
            l += str(0)
            l += '\n'
            f.write(l)

    t1 = time.time()

    print("")
    print("time {:.2f} count {:d} ".format(t1-t0, count))

    if args.seed == 1:
        tree_fn = save_dir + "/" + model_str + ".pkl"
        with open(tree_fn, 'wb') as f:
            pkl.dump(current_node, f)


# if __name__=='__main__':
#     max_score = [-99999.0, '']
#     count = 0
#     smiles = 'C=C(C)C'
#     state = State(smiles)
#     SIMULATION(state)
#     # node = UCTSEARCH(10, root)
#     # print("final best: ", node.state.smiles, node.reward)
#     # print("max score: ", max_score[1], max_score[0], Chem.MolFromSmiles(max_score[1]).GetNumAtoms())
