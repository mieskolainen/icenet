# Tool functions for B/RK analyzer  (protocode)
#
# [CODE IS ROTTEN / UNFIXED due to awkward0 --> awkward1 changes]
#
# NOTE: New branches (not originally inside nanoAOD tree) are identified with '_key'
#
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import copy
import uproot
import numpy as np
import networkx 
import itertools
import numba

import vector

from termcolor import colored

from icenet.tools import aux
from icebrk.PDG import *
import icebrk.features as features


def construct_kinematics(d, l1_p4, l2_p4, k_p4):
    """ Construct kinematics of the triplet @[JAGGED].

    Args:
        d:
        l1_p4:
        l2_p4:
        k_p4:

    Returns:
    """

    # Construct Awkward arrays from Awkward columns
    # Track 1
    l1_p4['e'] = vector.Array({'pt' : d['BToKEE_fit_l1_pt'],
                               'eta': d['BToKEE_fit_l1_eta'],
                               'phi': d['BToKEE_fit_l1_phi'],
                               'm': (d['BToKEE_fit_l1_pt'] - d['BToKEE_fit_l1_pt'] + 1) * ELECTRON_MASS})

    # Track 2
    l2_p4['e'] = vector.Array({'pt' : d['BToKEE_fit_l2_pt'],
                               'eta': d['BToKEE_fit_l2_eta'],
                               'phi': d['BToKEE_fit_l2_phi'],
                               'm': (d['BToKEE_fit_l2_pt'] - d['BToKEE_fit_l2_pt'] + 1) * ELECTRON_MASS})
    # Track 3
    k_p4['k']  = vector.Array({'pt': d['BToKEE_fit_k_pt'],
                               'eta': d['BToKEE_fit_k_eta'],
                               'phi': d['BToKEE_fit_k_phi'],
                               'm': (d['BToKEE_fit_k_pt'] - d['BToKEE_fit_k_pt'] + 1) * K_MASS})


def construct_MC_tree(d):
    """ Construct decay tree branches @[JAGGED].

    Args:
        d:

    Returns:
    """

    # Tracks
    d['_BToKEE_l1_genPdgId']       = d['GenPart_pdgId'][ d['Electron_genPartIdx'][ d['BToKEE_l1Idx']]  ]
    d['_BToKEE_l2_genPdgId']       = d['GenPart_pdgId'][ d['Electron_genPartIdx'][ d['BToKEE_l2Idx']]  ]
    d['_BToKEE_k_genPdgId']        = d['GenPart_pdgId'][ d['ProbeTracks_genPartIdx'][ d['BToKEE_kIdx']]]
    
    # Mothers
    d['_BToKEE_l1_genMotherIdx']   = d['GenPart_genPartIdxMother'][ d['Electron_genPartIdx'][ d['BToKEE_l1Idx']]  ]
    d['_BToKEE_l2_genMotherIdx']   = d['GenPart_genPartIdxMother'][ d['Electron_genPartIdx'][ d['BToKEE_l1Idx']]  ]
    d['_BToKEE_k_genMotherIdx']    = d['GenPart_genPartIdxMother'][ d['ProbeTracks_genPartIdx'][ d['BToKEE_kIdx']]]
    
    d['_BToKEE_l1_genMotherPdgId'] = d['GenPart_pdgId'][ d['_BToKEE_l1_genMotherIdx']]
    d['_BToKEE_l2_genMotherPdgId'] = d['GenPart_pdgId'][ d['_BToKEE_l2_genMotherIdx']]
    d['_BToKEE_k_genMotherPdgId']  = d['GenPart_pdgId'][ d['_BToKEE_k_genMotherIdx'] ]


def construct_MC_truth(d):
    """ Set MC signal truth into a new branch @[JAGGED].

    Args:
        d:

    Returns:
    """

    # Require track to be electron/positron and mother either B or J/psi
    VALUE1 = (abs(d['_BToKEE_l1_genPdgId']) == PDG_e) & ((abs(d['_BToKEE_l1_genMotherPdgId']) == PDG_B) | (abs(d['_BToKEE_l1_genMotherPdgId']) == PDG_JPSI))
    
    # Require track to be electron/positron and mother either B or J/psi
    VALUE2 = (abs(d['_BToKEE_l2_genPdgId']) == PDG_e) & ((abs(d['_BToKEE_l2_genMotherPdgId']) == PDG_B) | (abs(d['_BToKEE_l2_genMotherPdgId']) == PDG_JPSI))

    # Require track to be kaon and mother B
    VALUE3 = (abs(d['_BToKEE_k_genPdgId'])  == PDG_K) &  (abs(d['_BToKEE_k_genMotherPdgId'])  == PDG_B)

    # Finally, require same mother
    VALUE4 = (d['_BToKEE_l1_genMotherIdx'] == d['_BToKEE_l2_genMotherIdx'])
    
    # Create new branch
    d['_BToKEE_is_signal'] = (VALUE1 & VALUE2 & VALUE3 & VALUE4)


@numba.njit
def deltar_3(eta1,eta2,eta3, phi1,phi2,phi3, dR_MATCH):
    """ Match vector triplets by their DeltaR.

    Args:
    Returns:
    """

    qsets = []

    N = len(eta1)
    for i in range(N):
        this_set = [i]
        for j in range(N):
            if (j <= i): continue # No lower triangle needed

            # deltaR matching
            if ((aux.deltar(eta1[i],eta1[j], phi1[i],phi1[j]) < dR_MATCH) and
                (aux.deltar(eta2[i],eta2[j], phi2[i],phi2[j]) < dR_MATCH) and
                (aux.deltar(eta3[i],eta3[j], phi3[i],phi3[j]) < dR_MATCH)):

                this_set.append(j)
        qsets.append(this_set)

    return qsets


def find_connected_triplets(evt_index, l1_p4, l2_p4, k_p4, dR_MATCH = 0.01):
    """ Find all qsets of triplets connected together via DeltaR matching of their vectors.
    
    Args:
    Returns:
    """

    eta1 = l1_p4['e'][evt_index].eta # @JaggedArray acceleration
    eta2 = l2_p4['e'][evt_index].eta
    eta3 =  k_p4['k'][evt_index].eta

    phi1 = l1_p4['e'][evt_index].phi # @JaggedArray acceleration
    phi2 = l2_p4['e'][evt_index].phi
    phi3 =  k_p4['k'][evt_index].phi

    qsets = deltar_3(eta1,eta2,eta3, phi1,phi2,phi3, dR_MATCH)

    # Merge
    return aux.merge_connected(qsets)


def get_first_indices(qsets, MAXT3):
    """  Get the first evt_index from a list of list, where sublists
    encode e.g. different reconstruction chains of triplets.

    Args:
    Returns:
    """

    ind = np.zeros(MAXT3, dtype=int)
    k = 0
    for s in qsets:
        ind[k] = int(s[0]) # Get the first evt_index
        k += 1
        if k >= MAXT3: break
    return ind


def construct_input_vec(evt_index, d, l1_p4, l2_p4, k_p4, qsets, MAXT3):
    """ Construct MVA input vector x.

    feature 1 for all triplets 0 <possible zeros> .. 0 0,
    feature 2 for all triplets 0 <possible zeros> .. 0 0,
    
    
    feature D for all triplets 0 <possible zeros> .. 0 0]
    where zeros are padded after each feature if no enough triplets are found

    Args:
    Returns:
    """

    # Full input vector
    D = features.getdimension()
    x = np.zeros(D*MAXT3)

    # Loop over all MVA features
    z = 0

    for f in features.all_features.keys():
        vec = np.zeros(MAXT3) # Init with zeros!

        # Loop over supersets
        k = 0
        for s in qsets:
                    
            # Choose the active triplet of the superset (**** UPDATE THIS! ****)
            ind = s[0]

            vec[k] = np.array(d[f][evt_index][ind])                    # scalar
            k += 1

            if (k == MAXT3): break

        x[z*MAXT3 : (z+1)*MAXT3] = copy.deepcopy(vec)
        z += 1
    return x


def index_of_first_signal(evt_index, d, qsets, MAXT3):
    """ Check the evt_index of the last signal triplet (MC truth).
    
    Args:
    Returns:
    """
    first_index = -1
    k = 0
    for tset in qsets:
        for ind in tset: # Pick first of alternatives and break
            #[HERE ADD THE OPTION TO CHOOSE e.g. THE BEST RECONSTRUCTION QUALITY !!]
            y = np.asarray(d['_BToKEE_is_signal'][evt_index])[ind]
            break
        if y == 1:
            first_index = k
            break
        k += 1
    return first_index


def index_of_last_signal(evt_index, d, qsets, MAXT3):
    """ Check the evt_index of the last signal triplet (MC truth).
    
    Args:
    Returns:
    """
    last_index = -1
    k = 0
    for tset in qsets:
        for ind in tset: # Pick first of alternatives and break
            # [HERE ADD THE OPTION TO CHOOSE e.g. THE BEST RECONSTRUCTION QUALITY !!]
            y = np.asarray(d['_BToKEE_is_signal'][evt_index])[ind]
            break
        if y == 1:
            last_index = k
        k += 1
    return last_index


def construct_output_vec(evt_index, d, qsets, MAXT3):
    """ Construct MVA output vector y (binary with multilabel).
    
    Args:
    Returns:
    """

    y = np.zeros(MAXT3, dtype=int)
    k = 0
    for tset in qsets:
        for ind in tset: # Pick first of alternatives and break
            # [HERE ADD THE OPTION TO CHOOSE e.g. THE BEST RECONSTRUCTION QUALITY !!]
            y[k] = np.asarray(d['_BToKEE_is_signal'][evt_index])[ind]
            break
        k += 1
        if (k == MAXT3): break
    return y


def print_MC_event(evt_index, d, l1_p4, l2_p4, k_p4, qsets, PRINTMAX = 10000):
    """ Print MC event info.
    
    Args:
    Returns:
    """

    print('PDG ID: e (11), gamma (22), pi0 (111), pi (211), K(321), B (521)')
    print('notation: (pt,eta,phi) <low-pt electron | PF electron || overlap | conVeto> [G: generator PDG id (mother PDG id) ...]')

    def message(is_signal):
        if is_signal:
            return colored('S: TRUE', 'green')
        else:
            return colored('S: FALSE', 'red')

    B_mass = (l1_p4['e'] + l2_p4['e'] + k_p4['k']).mass
    k = 0
    for tset in qsets:
        print(colored(f'[{k}] = {tset}', 'magenta'))
        for i in tset:

            is_signal = d['_BToKEE_is_signal'][evt_index][i]
            print('  {:3} : [M(1+2+k): {:3.2} ({:5.2f},{:5.2f},{:5.2f}) <{}|{}||{}|{}> ({:5.2f},{:5.2f},{:5.2f}) <{}|{}||{}|{}> ({:5.2f},{:5.2f},{:5.2f})]  [SVp:{:6.2%} c2D:{:6.3f}]  [G:{:4}({:4}) {:4}({:4}) {:4}({:4})]'.
                format(i, B_mass[evt_index][i],
                    l1_p4['e'][evt_index][i].pt, l1_p4['e'][evt_index][i].eta, l1_p4['e'][evt_index][i].phi,
                    int(d['Electron_isLowPt'][d['BToKEE_l1Idx']][evt_index][i]), int(d['Electron_isPF'][d['BToKEE_l1Idx']][evt_index][i]), int(d['Electron_isPFoverlap'][d['BToKEE_l1Idx']][evt_index][i]), int(d['Electron_convVeto'][d['BToKEE_l1Idx']][evt_index][i]), 
                    l2_p4['e'][evt_index][i].pt, l2_p4['e'][evt_index][i].eta, l2_p4['e'][evt_index][i].phi,
                    int(d['Electron_isLowPt'][d['BToKEE_l2Idx']][evt_index][i]), int(d['Electron_isPF'][d['BToKEE_l2Idx']][evt_index][i]), int(d['Electron_isPFoverlap'][d['BToKEE_l2Idx']][evt_index][i]), int(d['Electron_convVeto'][d['BToKEE_l2Idx']][evt_index][i]),
                     k_p4['k'][evt_index][i].pt,  k_p4['k'][evt_index][i].eta,  k_p4['k'][evt_index][i].phi,
                        d['BToKEE_svprob'][evt_index][i],       d['BToKEE_fit_cos2D'][evt_index][i],
                        d['_BToKEE_l1_genPdgId'][evt_index][i], d['_BToKEE_l1_genMotherPdgId'][evt_index][i],
                        d['_BToKEE_l2_genPdgId'][evt_index][i], d['_BToKEE_l2_genMotherPdgId'][evt_index][i],
                        d['_BToKEE_k_genPdgId'][evt_index][i] , d['_BToKEE_k_genMotherPdgId'][evt_index][i] ,
                        ), message(is_signal))

            k += 1
            if (k == PRINTMAX):
                print('')
                return
