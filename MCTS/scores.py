from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

from rdkit.Chem import QED
from score_modules.ESOL_Score.esol import ESOLCalculator
from score_modules.SA_Score import sascorer
#from score_modules.RGES_Score.rges import RGESCalculator
from score_modules.GSK3B_Score.gsk3b import GSK3BCalculator
from score_modules.JNK3_Score.jnk3 import JNK3Calculator
from score_modules.COVID_Score.covid import COVIDCalculator


esol_calculator = ESOLCalculator()
#rges_calculator = RGESCalculator()
gsk3b_calculator = GSK3BCalculator()
jnk3_calculator = JNK3Calculator()
covid_calculator = COVIDCalculator()


def gsk3b(mol):
    return gsk3b_calculator.get_score(mol)


def jnk3(mol):
    return jnk3_calculator.get_score(mol)


def qed(mol):
    return QED.qed(mol)


def sa(mol):
    sa_score = sascorer.calculateScore(mol)
    normalized_sa = (10. - sa_score) / 9.
    return normalized_sa


def npc1(mol):
    return covid_calculator.get_score(mol, 'npc1')


def insig1(mol):
    return covid_calculator.get_score(mol, 'insig1')


def hmgcs1(mol):
    return covid_calculator.get_score(mol, 'hmgcs1')



def esol(mol):
    return esol_calculator.calc_esol(mol)


def rges(mol):
    return -1 * rges_calculator.rges_score(mol)


def get_largest_ring_size(mol):
    cycle_list = mol.GetRingInfo().AtomRings()
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length


def plogp(mol):
    log_p = Descriptors.MolLogP(mol)
    sas_score = sascorer.calculateScore(mol)
    largest_ring_size = get_largest_ring_size(mol)
    cycle_score = max(largest_ring_size - 6, 0)
    p_logp = log_p - sas_score - cycle_score
    return p_logp


def qed_sa(mol):
    qed_score = qed(mol)
    sa_score = sa(mol)
    nomalized_sa = (sa_score-1)/9.0
    return qed_score+sa_score


if __name__=='__main__':
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    import pandas as pd

    df = pd.read_csv('libs/drug_like_cluster_info.csv')
    smiles = df['smiles'].tolist()
    plogp_scores = []
    qed_scores = []
    sa_scores = []
    rges_scores = []
    esol_scores = []
    gsk3b_scores = []
    jnk3_scores = []
    i = 0
    for s in smiles:
        if i%100 == 0:
            print(i)
        mol = Chem.MolFromSmiles(s)
        plogp_scores.append(plogp(mol))
        qed_scores.append(qed(mol))
        sa_scores.append(sa(mol))
        rges_scores.append(rges(mol))
        esol_scores.append(esol(mol))
        gsk3b_scores.append(gsk3b(mol))
        jnk3_scores.append(jnk3(mol))
        i+=1

    df['plogp'] = plogp_scores
    df['qed'] = qed_scores
    df['sa'] = sa_scores
    df['gsk3b'] = gsk3b_scores
    df['jnk3'] = jnk3_scores
    df['rges'] = rges_scores
    df['esol'] = esol_scores
    df.to_csv('libs/drug_like_cluster_prop_info.csv', index=False)




