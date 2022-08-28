import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class GSK3BCalculator():

    def __init__(self, **kwargs):
        super(GSK3BCalculator, self).__init__(**kwargs)
        self.model = self._load_model()

    def _load_model(self):
	    with open('score_modules/GSK3B_Score/gsk3b.pkl', 'rb') as f:	#score_modules/GSK3B_Score/
	        model =  pickle.load(f, encoding='iso-8859-1')
	    return model

    def _get_morgan_fingerprint(self, mol, radius=2, nBits=1024):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useFeatures=False)
        fp_bits = fp.ToBitString()
        #finger_print = np.fromstring(fp_bits, 'u1')-ord('0')
        finger_print = np.array(list(map(int, fp_bits)))
        return finger_print

    def get_score(self, mol):
        fp = self._get_morgan_fingerprint(mol).astype(np.float32).reshape(1, -1)
        score = self.model.predict_proba(fp)[0, 1]
        return score


if __name__ == '__main__':
    s = 'C1=CC(=C(C=C1[N+](=O)[O-])Cl)NC(=O)C2=C(C=CC(=C2)Cl)O'
    mol = Chem.MolFromSmiles(s)
    gsk3b_calculator=GSK3BCalculator()
    print(gsk3b_calculator.get_score(mol))
