# mmpdb - matched molecular pair database generation and analysis
#
# Copyright (c) 2015-2017, F. Hoffmann-La Roche Ltd.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
#    * Neither the name of F. Hoffmann-La Roche Ltd. nor the names of
#      its contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from __future__ import print_function

import sys
import time
import multiprocessing
import numpy as np
from rdkit import Chem

from mmpdblib import command_support
from mmpdblib import dbutils
from mmpdblib import cpdGeneration_algorithm_optimized

from mmpdblib import fileio
from mmpdblib.cpdGeneration_algorithm_optimized import open_database

import pandas as pd
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

import argparse

from mmpdblib import smarts_aliases
from mmpdblib.config import nonnegative_int, cutoff_list, positive_int
from mmpdblib.cpdGeneration_algorithm_optimized import open_database

########################
def add_diff_heavies(dataFrame):
    '''
    Compute the difference between new and old fragment
    and update the dataframe
    '''
    diff = []
    for idx, row in dataFrame.iterrows():
        f1 = row.original_frag
        f2 = row.new_frag
        f1 = Chem.MolFromSmiles(f1)
        f2 = Chem.MolFromSmiles(f2)

        hac1 = 0
        hac2 = 0
        if f1 is not None:
            hac1 = f1.GetNumHeavyAtoms()
        if f2 is not None:
            hac1 = f2.GetNumHeavyAtoms()
        diff.append(np.abs(hac1 - hac2))

    dataFrame.insert(len(dataFrame.columns), "heavies_diff", diff)
    return dataFrame

def remove_star_atom(mol):
    for atom in mol.GetAtoms():
         if atom.GetSymbol() == "*" or atom.GetSymbol() == "[*]":
             atom.SetAtomicNum(2)
    mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts('[He]'))
    return mol


def is_containSubstructure(mol, patt):
    if mol is None or patt is None:
        return True
    mol = remove_star_atom(mol)
    patt = remove_star_atom(patt)
    return mol.HasSubstructMatch(patt)
    

########################
def get_fragmentAtomMapping(replaceGroup, qmolecule):
    '''
    Mapped the replaceGroup to query molecule and returns
    corresponding atom indexes from query molecule
    replaceGroup: smiles notation of replaceGroup
    qmolecule: smiles notation of complete query molecule
    '''

    replaceMol = Chem.MolFromSmarts(replaceGroup)
    qmol = Chem.MolFromSmiles(qmolecule, sanitize=False)
    matches = list(qmol.GetSubstructMatches(replaceMol))

    if len(matches) == 0:
        return matches

    # ommit the * atom index from list
    # logic is that: atom mapping order result from qmol
    # equals to order of atoms in replaceGroup. Which means:
    # if * atom is having index 5 in replaceGroup
    # the corresponding matching atom from query molecule will be
    # at 6th position in result list.
    filter_matches = []
    for mat in matches:
        for i in range(len(mat)):
            if replaceMol.GetAtomWithIdx(i).GetSymbol() == "*":
                continue
            else:
                filter_matches.append(mat[i])
    return filter_matches


########################
# Helper function to make a new function which format time deltas
# so all the "."s are lined up and they use the minimal amount of
# left-padding.
def get_time_delta_formatter(max_dt):
    s = "%.1f" % (max_dt,)
    num_digits = len(s)
    fmt = "%" + str(num_digits) + ".1f"

    def format_dt(dt):
        return fmt % (dt,)

    return format_dt


########################
class DesignMove():

    def __init__(self, db_path):
        self.db_connection = open_database(db_path)
        # default args
        self.radius = 3
        self.min_variable_size = 0
        self.max_variable_size = 9999
        self.min_constant_size = 0
        self.tconstant_smi = None
        # self.args = self._generate_constant_args()
        alias_names = ", ".join(repr(alias.name) for alias in smarts_aliases.cut_smarts_aliases)
        default_configs = {"tsubstructure": None, 
                            "tjobs": 1, 
                            "toutput":None,
                            "ttimes":True,
                            "replaceGroup":None,
                            "tmax_heavies":100,
                            "tmax_rotatable_bonds":40,
                            "trotatable_smarts":"[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]",
                            "tsalt_remover":"<default>",
                            "tcut_smarts":smarts_aliases.cut_smarts_aliases_by_name["default"].smarts,
                            "tnum_cuts":3,
                            "tmin_heavies_per_const_frag":0,
                            "tmin_heavies_total_const_frag":0
                            }
        self.args = argparse.Namespace(**default_configs)
   
    def _generate_constant_args(self):
        p = argparse.ArgumentParser(description='Default args for design move.')
        p.add_argument("--tsubstructure", metavar="SMARTS",
                       help="require the substructure pattern in the product")
        p.add_argument("--tjobs", type=positive_int, default=1,
                       help="number of jobs to run in parallel (default: 1)")
        p.add_argument("--toutput", metavar="FILENAME", default=None,
                       help="save the output to FILENAME (default=stdout)")
        p.add_argument("--ttimes", action="store_true",
                       help="report timing information for each step")
        p.add_argument("--replaceGroup", required=False, default=None,
                       help="a smiles (with attachments points as start) indicating as which group to replace")
        # fragmentation option
        p.add_argument("--tmax-heavies", type=nonnegative_int, metavar="N", default=100,
                       help="Maximum number of non-hydrogen atoms")

        p.add_argument("--tmax-rotatable-bonds", type=nonnegative_int, metavar="N", default=40, # original 10, increase to 20 to avoid error in fragmentation
                       help="Maximum number of rotatable bonds")

        p.add_argument("--trotatable-smarts", metavar="SMARTS",
                       default="[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]",
                       help="SMARTS pattern to detect rotatable bonds (default: %r)")

        p.add_argument("--tsalt-remover", metavar="FILENAME", choices=["<none>", "<default>"],
                       help="File containing RDKit SaltRemover definitions. The default ('<default>') "
                            "uses RDKit's standard salt remover. Use '<none>' to not remove salts.", default="<default>")

        alias_names = ", ".join(repr(alias.name) for alias in smarts_aliases.cut_smarts_aliases)
        p.add_argument("--tcut-smarts", metavar="SMARTS", default=smarts_aliases.cut_smarts_aliases_by_name["default"].smarts,
                       help="alternate SMARTS pattern to use for cutting")

        p.add_argument("--tnum-cuts", choices=(1, 2, 3), type=nonnegative_int,
                       help="number of cuts to use", default=3)

        p.add_argument("--tmin-heavies-per-const-frag", type=nonnegative_int,
                       metavar="N", default=0,
                       help="Ignore fragmentations where one or more constant fragments are very small")

        p.add_argument("--tmin-heavies-total-const-frag", type=nonnegative_int,
                       metavar="N", default=0,
                       help="Ignore fragmentations where sum of constant fragments are very small")
        return p.parse_args()

    def one_step_move(self, query_smi, min_pairs):
        fragId_to_fragsmi = cpdGeneration_algorithm_optimized.read_fragmentIndex_to_smiTable(self.db_connection)
        envsmiId_to_envsmi = cpdGeneration_algorithm_optimized.read_envsmiId_to_envsmiTable(self.db_connection)

        substructure_pat = None

        # get the transform tool
        transform_tool = cpdGeneration_algorithm_optimized.get_transform_tool(self.db_connection, self.args)

        # output data-frame
        output_df = None

        # query smiles
        smi = query_smi

        # fragment entire compound
        transform_record = transform_tool.fragment_transform_smiles(smi)
        transform_record = transform_tool.expand_variable_symmetry(transform_record)

        if transform_record.errmsg:
            sys.stdout.write("ERROR: Unable to fragment --smiles %r: %s" % (query_smi, transform_record.errmsg))
            exit()

        # Check if the replace-group fragment specified by users is
        # present in fragmentation of entire molecule or not
        replaceGroup_Found = False
        replaceGroup_Mol = None
        fragments = None

        # Three possible scenes for fragmentation's:
        # 1) if replace group is not provided i.e. None then consider all fragmentation's
        # 2) if replace group is single atom then consider that specific fragmentation
        # 3) if replace group is more than one atom, then re-fragment replaceGroup and
        # consider all those fragments as queries
        possible_frags = {}
        original_constatPart_as_mol = None

        fragments = transform_record.fragments

        # filter the fragments to restrict to the ones, whose constant part is present in original constant part
        fragments_filter = []
        for frag in fragments:
            if original_constatPart_as_mol is not None and self.tconstant_smi.count("*") == 1:
                constantPart = Chem.MolFromSmiles(frag.constant_smiles, sanitize=True)
                is_pass = is_containSubstructure(constantPart, original_constatPart_as_mol)
                if is_pass:
                    fragments_filter.append(frag)
                else:
                    continue
            else:
                fragments_filter.append(frag)


        try:
            pool = multiprocessing.Pool(processes=self.args.tjobs)
            output_df = transform_tool.transform(fragments_filter,
                                                 radius=self.radius,
                                                 min_pairs=min_pairs,
                                                 min_variable_size=self.min_variable_size,
                                                 max_variable_size=self.max_variable_size,
                                                 min_constant_size=self.min_constant_size,
                                                 substructure_pat=substructure_pat,
                                                 pool=pool, db_fragId_to_fragsmi=fragId_to_fragsmi,
                                                 db_envsmiId_to_envsmi=envsmiId_to_envsmi
                                                 )
            pool.close()    # close pool to avoid too many open files
            
            # add the original smiles to data-frame
            smis = [smi] * len(output_df)
            output_df.insert(0, "original_smi", smis)
            
            if len(output_df) == 0:
                sys.stdout.write("ERROR: Everything was good, but no rules were found")
                exit()

            # sort based on freq
            output_df = output_df.sort_values(["rule_freq"], ascending=False)
            output_df.reset_index(inplace=True, drop=True)

        except cpdGeneration_algorithm_optimized.EvalError as err:
            sys.stdout.write("ERROR: %s\nExiting.\n" % (err,))
            exit()

        # add diff heavies column
        output_df = add_diff_heavies(output_df)
        
        # remove duplicates
        output_df = output_df.drop_duplicates(["transformed_smi"])
        output_df.reset_index(inplace=True, drop=True)


        return output_df["transformed_smi"].tolist()


   

def design_move_wrapper(db_connection, query_smi, min_pairs, args):
    # Generator
    radius = str(3)
    assert radius in list("012345"), radius
    radius = int(radius)
    min_pairs = min_pairs
    min_variable_size = 0
    max_variable_size = 9999
    assert max_variable_size > min_variable_size, "max-variable-size must be greater than min-variable-size"
    min_constant_size = 0
    args.tconstant_smi = None

    # I preferred to do this, as querying several times this database is seems to be slow
    # If you make one big query and execute it, then you will encounter other sqlite runtime error
    # for instance max. tree dept limit
    # read fragment index to fragment smi table
    # Off course there is work around, but its like more coding for low performance gain
    fragId_to_fragsmi = cpdGeneration_algorithm_optimized.read_fragmentIndex_to_smiTable(db_connection)

    # read envsmi index to envsmi table
    envsmiId_to_envsmi = cpdGeneration_algorithm_optimized.read_envsmiId_to_envsmiTable(db_connection)

    substructure_pat = None

    # get the transform tool
    print(args)
    transform_tool = cpdGeneration_algorithm_optimized.get_transform_tool(db_connection, args)

    # output data-frame
    output_df = None

    # query smiles
    smi = query_smi

    # fragment entire compound
    transform_record = transform_tool.fragment_transform_smiles(smi)
    transform_record = transform_tool.expand_variable_symmetry(transform_record)

    if transform_record.errmsg:
        sys.stdout.write("ERROR: Unable to fragment --smiles %r: %s" % (query_smi, transform_record.errmsg))
        exit()

    # Check if the replace-group fragment specified by users is
    # present in fragmentation of entire molecule or not
    replaceGroup_Found = False
    replaceGroup_Mol = None
    fragments = None

    # Three possible scenes for fragmentation's:
    # 1) if replace group is not provided i.e. None then consider all fragmentation's
    # 2) if replace group is single atom then consider that specific fragmentation
    # 3) if replace group is more than one atom, then re-fragment replaceGroup and
    # consider all those fragments as queries
    possible_frags = {}
    original_constatPart_as_mol = None

    fragments = transform_record.fragments

    # filter the fragments to restrict to the ones, whose constant part is present in original constant part
    fragments_filter = []
    for frag in fragments:
        if original_constatPart_as_mol is not None and args.tconstant_smi.count("*") == 1:
            constantPart = Chem.MolFromSmiles(frag.constant_smiles, sanitize=True)
            is_pass = is_containSubstructure(constantPart, original_constatPart_as_mol)
            if is_pass:
                fragments_filter.append(frag)
            else:
                continue
        else:
            fragments_filter.append(frag)


    try:
        pool = multiprocessing.Pool(processes=1)
        output_df = transform_tool.transform(fragments_filter,
                                             radius=radius,
                                             min_pairs=min_pairs,
                                             min_variable_size=min_variable_size,
                                             max_variable_size=max_variable_size,
                                             min_constant_size=min_constant_size,
                                             substructure_pat=substructure_pat,
                                             pool=pool, db_fragId_to_fragsmi=fragId_to_fragsmi,
                                             db_envsmiId_to_envsmi=envsmiId_to_envsmi
                                             )

        # add the original smiles to data-frame
        smis = [smi] * len(output_df)
        output_df.insert(0, "original_smi", smis)
        
        if len(output_df) == 0:
            sys.stdout.write("ERROR: Everything was good, but no rules were found")
            exit()

        # sort based on freq
        output_df = output_df.sort_values(["rule_freq"], ascending=False)
        output_df.reset_index(inplace=True, drop=True)

    except cpdGeneration_algorithm_optimized.EvalError as err:
        sys.stdout.write("ERROR: %s\nExiting.\n" % (err,))
        exit()

    # add diff heavies column
    output_df = add_diff_heavies(output_df)
    
    # remove duplicates
    output_df = output_df.drop_duplicates(["transformed_smi"])
    output_df.reset_index(inplace=True, drop=True)

    return output_df["transformed_smi"].tolist()
########################

if __name__ == "__main__":
    replaceRule = DesignMove("chemblDB3.sqlitdb")
    s = 'C1=CC(=C(C=C1[N+](=O)[O-])Cl)NC(=O)C2=C(C=CC(=C2)Cl)O'
    output = replaceRule.one_step_move(query_smi=s, min_pairs=100)
    print(output)
