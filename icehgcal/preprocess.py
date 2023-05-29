# HGCAL hackathon trackster tree processor
# 
# "/eos/cms/store/group/dpg_hgcal/comm_hgcal/hackathon/samples/close_by_double_pion/production/new_new_ntuples/"
#
# m.mieskolainen@imperial.ac.uk, 2023

import awkward as ak
import numpy as np
import uproot as uproot
import numba
from tqdm import tqdm


@numba.jit
def compute_edges(trk_data, ass_data, gra_data, edge_index, edge_labels, edge_qualities, graph_param):
    """
    Logic based on (but refined):
    https://github.com/Abhirikshma/HackathonLinking/blob/master/firstModelAndTraining.ipynb
    
    Returns:
        node, edge, edge_labels
    """

    # From yaml file
    thresh     = graph_param['thresh']
    directed   = graph_param['directed']
    self_loops = graph_param['self_loops']
    
    # Each trackster ~ node in the ML-graph
    for i in range(trk_data.NTracksters):

        qualities_i = ass_data.tsCLUE3D_recoToSim_CP_score[i]
        best_sts_i  = ass_data.tsCLUE3D_recoToSim_CP[i][ak.argmin(qualities_i)]
        OK_i        = True if qualities_i[best_sts_i] < thresh else False

        # Found within (eta,phi) window, presumably
        for j in gra_data.linked_inners[i]:

            qualities_j = ass_data.tsCLUE3D_recoToSim_CP_score[j]
            best_sts_j  = ass_data.tsCLUE3D_recoToSim_CP[j][ak.argmin(qualities_j)]
            OK_j        = True if qualities_j[best_sts_j] < thresh else False

            # -----------------------------
            ## Create ML-graph data
            
             # ** MC truth **
            if (best_sts_i == best_sts_j) and (OK_i and OK_j):
                edge_labels.append(1)
            else:
                edge_labels.append(0)

            edge_index.append([i,j])
            edge_qualities.append([qualities_i[best_sts_i], qualities_j[best_sts_j]])
            
            # If we want undirected graph
            if directed:
                edge_labels.append(edge_labels[-1])
                edge_index.append([j,i])
                edge_qualities.append([qualities_j[best_sts_j], qualities_i[best_sts_i]])
            # -----------------------------

        # If we want self-loops
        if self_loops:
            edge_labels.append(1)
            edge_index.append([i,i])
            edge_qualities.append([qualities_i[best_sts_i], qualities_i[best_sts_i]])


def create_trackster_data(files):
    
    calos        = []
    tracksters   = []
    associations = []
    graph        = []

    for file in files:

        try:
            print('.', end = "")
            
            f    = uproot.open(file)
            t    = f["ntuplizer/tracksters"]
            calo = f["ntuplizer/simtrackstersCP"]
            ass  = f["ntuplizer/associations"]
            gra  = f["ntuplizer/graph"]

            # Tracksters
            tracksters.append(t.arrays(["NTracksters",
                                        "raw_energy", 
                                        "raw_em_energy",
                                        "trackster_barycenter_eta", 
                                        "trackster_barycenter_phi",
                                        "barycenter_x", 
                                        "barycenter_y", 
                                        "barycenter_z", 
                                        "id_probabilities",
                                        "EV1",
                                        "EV2",
                                        "EV3",
                                        "eVector0_x",
                                        "eVector0_y", 
                                        "eVector0_z",
                                        "sigmaPCA1",
                                        "sigmaPCA2",
                                        "sigmaPCA3"]))
            # Calorimetry
            calos.append(calo.arrays(["stsCP_trackster_barycenter_eta",
                          "stsCP_trackster_barycenter_phi",
                          "stsCP_barycenter_x", 
                          "stsCP_barycenter_y", 
                          "stsCP_barycenter_z",
                          "stsCP_raw_energy"]))

            # Associations
            associations.append(ass.arrays(["tsCLUE3D_recoToSim_CP", "tsCLUE3D_recoToSim_CP_score"]))
            
            # Links
            graph.append(gra.arrays(["linked_inners"]))

        except:
            print(f"Error in {file}")

    df_calo  = ak.concatenate(calos)
    df_track = ak.concatenate(tracksters)
    df_ass   = ak.concatenate(associations)
    df_gra   = ak.concatenate(graph)

    return {'df_calo': df_calo, 'df_track': df_track, 'df_ass': df_ass, 'df_gra': df_gra}


def event_loop(files, graph_param, maxevents=int(1E9)):

    # --------------------------------------------

    # Create trackster data
    data  = create_trackster_data(files=files)

    x              = []
    edge_index     = []
    edge_labels    = []
    edge_qualities = []

    N = np.min([maxevents, len(data['df_track'])])

    for ev in tqdm(range(N)):
        
        trk_data = data['df_track'][ev]
        gra_data = data['df_gra'][ev]
        ass_data = data['df_ass'][ev]

        # Compute node data
        x_ = ak.zip({'raw_energy':    trk_data.raw_energy, 
                     'raw_em_energy': trk_data.raw_em_energy,
                     'barycenter_x':  trk_data.barycenter_x,
                     'barycenter_y':  trk_data.barycenter_y,
                     'barycenter_z':  trk_data.barycenter_z,
                     'EV1':           trk_data.EV1,
                     'EV2':           trk_data.EV2,
                     'EV3':           trk_data.EV3
        })

        edge_index_     = []
        edge_labels_    = []
        edge_qualities_ = []

        # Compute edge data and labels
        compute_edges(trk_data=trk_data, ass_data=ass_data, gra_data=gra_data,
            edge_index=edge_index_, edge_labels=edge_labels_, edge_qualities=edge_qualities_, graph_param=graph_param)
        
        # Save event data
        x.append(x_)
        edge_labels.append(edge_labels_)
        edge_index.append(np.array(edge_index_).T)
        edge_qualities.append(np.array(edge_qualities_).T)
    
    return {'x': x, 'edge_index': edge_index, 'edge_labels': edge_labels, 'edge_qualities': edge_qualities}
