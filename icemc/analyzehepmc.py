# A tool to verify HepMC(2) event content properties for DQCD analysis
#
# DQCD scenarios:     https://arxiv.org/pdf/2303.04167.pdf
# HepMC status codes: https://arxiv.org/pdf/1912.08005.pdf (Table 5)
# PDG code scheme:    https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
# PDG MC listing:     https://github.com/mieskolainen/graniitti/blob/master/modeldata/mass_width_2020.mcd 
#
# Run with: python icemc/analyzehepmc.py 
# 
# m.mieskolainen@imperial.ac.uk, 2023

import matplotlib.pyplot as plt
import pyhepmc
import numpy as np
import os
import PyPDF2
from termcolor import colored, cprint

PDG_ETA3   = 4900221 # neutral dark "eta"
PDG_RHO3   = 4900113 # neutral dark "rho"
PDG_PI3    = 4900111 # neutral dark "pion"
PDG_APRIME = 999999  # dark "photon"
PDG_MUON   = 13


def beta(p):
    """
    Lorentz beta factor
    """
    return p.p3mod() / p.e

def gamma(p):
    """
    Lorentz gamma factor
    """
    return p.e / p.m()


def event_processor_contact_topology(filename, prop, dark_meson_name, dark_meson_PDG, eprint: int=100, print_first=True):
    """
    Process event content for the contact decay: dark meson -> mu+mu- [+ ...]
    """
    with pyhepmc.open(filename) as f:
        
        ev = 0
        for event in f:
            if ev == 0 and print_first:
                print(event)
                print("")
            
            for v in event.vertices:
                
                # dark meson -> mu+mu- vertex
                if any(q.pid == dark_meson_PDG for q in v.particles_in) and any(abs(q.pid) == PDG_MUON for q in v.particles_out):
                    
                    # Vertex and dark meson momentum
                    vertex = v.position
                    mom    = v.particles_in[0].momentum
                    
                    dxy  = vertex.perp()    # wrt. origin (0,0,0), access components with vertex.x etc.
                    d3   = vertex.p3mod()
                    
                    # decay length in lab = c*tau*beta*gamma, where tau the lifetime in the rest frame
                    ctau = d3 / (beta(mom) * gamma(mom))
                    
                    prop[f"vertex({dark_meson_name}->mu+mu-)"]["dxy_0"].append(dxy)
                    prop[f"vertex({dark_meson_name}->mu+mu-)"]["d3_0"].append(d3)
                    
                    prop[f"momentum({dark_meson_name})"]["ctau"].append(ctau)
                    prop[f"momentum({dark_meson_name})"]["pt"].append(mom.pt())
                    prop[f"momentum({dark_meson_name})"]["eta"].append(mom.eta())
                    prop[f"momentum({dark_meson_name})"]["m"].append(mom.m())
                    
                    # Muon (+) momentum
                    for i in range(len(v.particles_out)):
                        
                        if v.particles_out[i].pid == PDG_MUON: 
                            
                            mom = v.particles_out[i].momentum
                            
                            prop["momentum(mu+)"]["pt"].append(mom.pt())
                            prop["momentum(mu+)"]["eta"].append(mom.eta())
                            prop["momentum(mu+)"]["status"].append(v.particles_out[i].status)
                            
                            break

                    if eprint is not None and (ev % eprint == 0):
                        cprint(f"{dark_meson_name} -> mu+mu- vertex (event = {ev}) [{filename}]", "yellow")
                        print(f"vertex: {v}")
                        print(f"in:     {v.particles_in}")
                        print(f"out:    {v.particles_out}")
                        print("")
                    
            ev += 1
    
        cprint(f'cross-section: {event.cross_section.xsec():0.5f} +- {event.cross_section.xsec_err():0.5f} pb', 'yellow')
        print("")
    
    return prop


def event_processor_sequential_topology(filename, prop, dark_meson_name, dark_meson_PDG, eprint: int=100, print_first=True):
    """
    Process event content for the sequential decay: dark meson -> A' (-> mu+mu-) [+ ...]
    """
    with pyhepmc.open(filename) as f:
        ev = 0
        for event in f:
            if ev == 0 and print_first:
                print(event)
                print("")
            
            for v in event.vertices:
                
                # dark meson -> A'A' vertex
                if any(q.pid == dark_meson_PDG for q in v.particles_in) and any(q.pid == PDG_APRIME for q in v.particles_out):
                    
                    # Vertex and momentum of the dark meson
                    vertex = v.position
                    mom    = v.particles_in[0].momentum
                    
                    # decay length in lab = c*tau*beta*gamma, where tau the lifetime in the rest frame
                    ctau = vertex.p3mod() / (beta(mom) * gamma(mom))
                    
                    prop[f"vertex({dark_meson_name}->A'A')"]["dxy_0"].append(vertex.perp()) # wrt. origin (0,0,0)
                    prop[f"vertex({dark_meson_name}->A'A')"]["d3_0"].append(vertex.p3mod())
                    
                    prop[f"momentum({dark_meson_name})"]["ctau"].append(ctau)
                    prop[f"momentum({dark_meson_name})"]["pt"].append(mom.pt())
                    prop[f"momentum({dark_meson_name})"]["eta"].append(mom.eta())
                    prop[f"momentum({dark_meson_name})"]["m"].append(mom.m())
                    
                    if eprint is not None and (ev % eprint == 0):
                        cprint(f"{dark_meson_name} -> A'A' vertex (event = {ev}) [{filename}]", "yellow")
                        print(f"vertex: {v}")
                        print(f"in:     {v.particles_in}")
                        print(f"out:    {v.particles_out}")
                        print("")
            
            
            for v in event.vertices:
            
                # A' -> mu+mu-
                if any(q.pid == PDG_APRIME for q in v.particles_in) and any(abs(q.pid) == PDG_MUON for q in v.particles_out):
                    
                    # Vertex and A' momentum
                    vertex_A = v.position
                    mom_A    = v.particles_in[0].momentum
                    
                    prop["vertex(A'->mu+mu-)"]["dxy_0"].append(vertex_A.perp()) # wrt. origin (0,0,0)
                    prop["vertex(A'->mu+mu-)"]["d3_0"].append(vertex_A.p3mod())
                    
                    # -----------------------------------------------------
                    # Get parent (i.e. dark meson) vertex position
                    vertex_dark_meson = v.particles_in[0].production_vertex.position
                    
                    ## Decay length vector
                    delta = vertex_A - vertex_dark_meson

                    dxy  = delta.perp()
                    d3   = delta.p3mod()
                    ctau = d3 / (beta(mom_A) * gamma(mom_A))
                    
                    # -----------------------------------------------------
                    
                    prop["vertex(A'->mu+mu-)"]["dxy"].append(dxy)
                    prop["vertex(A'->mu+mu-)"]["d3"].append(d3)
                    
                    prop["momentum(A')"]["ctau"].append(ctau)
                    prop["momentum(A')"]["pt"].append(mom_A.pt())
                    prop["momentum(A')"]["eta"].append(mom_A.eta())
                    prop["momentum(A')"]["m"].append(mom_A.m())
                    
                    # Muon (+) momentum
                    for i in range(len(v.particles_out)):
                        
                        if v.particles_out[i].pid == PDG_MUON: 
                            
                            mom = v.particles_out[i].momentum
                            
                            prop["momentum(mu+)"]["pt"].append(mom.pt())
                            prop["momentum(mu+)"]["eta"].append(mom.eta())
                            prop["momentum(mu+)"]["status"].append(v.particles_out[i].status)
                            
                            break
                    
                    if eprint is not None and (ev % eprint == 0):
                        cprint(f"A' -> mu+mu- vertex (event = {ev}) [{filename}]", "yellow")
                        print(f"vertex: {v}")
                        print(f"in:     {v.particles_in}")
                        print(f"out:    {v.particles_out}")
                        print("")
                    
            ev += 1

        cprint(f'cross-section: {event.cross_section.xsec():0.5f} +- {event.cross_section.xsec_err():0.5f} pb', 'yellow')
        print("")
    
    return prop


def compute_stats(prop, savename, outpath):
    """
    Compute histograms and point statistics
    """
    
    cprint(f"----------------------------------------------------", "yellow")
    cprint(f"Sample statistics: {savename}", "yellow")
    cprint(f"----------------------------------------------------", "yellow")
    
    
    # Delete old plots
    cmd = f'rm -f {outpath}/*.pdf'
    cprint(cmd, 'red')
    os.system(cmd)
    
    for key in prop.keys():
        print("")
        for var in prop[key].keys():
            
            x = prop[key][var]
            
            # Point statistics
            print(f"{key}: [{var}] mean = {np.mean(x)}, std = {np.std(x)}")

            # Histogram
            fig, ax = plt.subplots(1,2)
            fig.tight_layout() # space between subplots
            
            for i in range(2):
                plt.sca(ax[i])
                
                if i == 0:
                    plt.hist(x, 100, label=f'min = {np.min(x):0.2e}, max = {np.max(x):0.2e}')
                    plt.title(f'{savename} | {key}', color=(0.5, 0, 0))
                    plt.ylabel('counts')
                    plt.legend(fontsize=7)
                
                if i == 1:
                    plt.hist(x, 100, label=f'$\\mu = {np.mean(x):0.2e}, \\sigma = {np.std(x):0.2e}$')
                    plt.yscale('log')
                    plt.legend(fontsize=7)

                plt.xlabel(f'{var}')
                
                ax[i].set_box_aspect(0.75)
            
            filename = f'{outpath}/{savename}_{key}_{var}.pdf'
            plt.savefig(filename, bbox_inches='tight')
            cprint(f'Saved figure: {filename}', "yellow")
            plt.close()

    # Merge pdfs
    pdf_merger(path=outpath, savename=savename)


def wrapper_contact_topology(p, scenario, basepath, dark_meson_name, dark_meson_PDG):
    """
    Wrapper function
    """
    
    CWD = os.getcwd()
    
    prop = {
        f"vertex({dark_meson_name}->mu+mu-)": {
            "dxy_0": [],
            "d3_0" : []
        },
        f"momentum({dark_meson_name})": {
            "pt" :   [],
            "eta":   [],
            "m"  :   [],
            "ctau" : []
        },
        "momentum(mu+)": {
            "pt" :  [],
            "eta":  [],
            "status": []
        }
    }

    # Loop over files
    for id in p['file_id']:
        filename = f"{basepath}/{scenario}/{scenario}_{id}.hepmc"
        prop     = event_processor_contact_topology(filename=filename, prop=prop,
                                              dark_meson_name=dark_meson_name, dark_meson_PDG=dark_meson_PDG)

    OUT_PATH = f'{CWD}/figs/analyzehepmc/{scenario}'
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    
    compute_stats(prop=prop, savename=f'{scenario}', outpath=OUT_PATH)


def wrapper_sequential_topology(p, scenario, basepath, dark_meson_name, dark_meson_PDG):
    """
    Wrapper function
    """
    
    CWD = os.getcwd()
    
    prop = {
        f"vertex({dark_meson_name}->A'A')": {
            "dxy_0" : [],
            "d3_0"  : []
        },
        "vertex(A'->mu+mu-)": {
            "dxy_0" : [],
            "d3_0"  : [],
            "dxy"   : [],
            "d3"    : []
        },
        f"momentum({dark_meson_name})": {
            "pt":   [],
            "eta":  [],
            "m":    [],
            "ctau": []
        },
        "momentum(A')": {
            "pt":   [],
            "eta":  [],
            "m":    [],
            "ctau": []
        },
        "momentum(mu+)": {
            "pt":     [],
            "eta":    [],
            "status": []
        }
    }

    # Loop over files
    for id in p['file_id']:
        filename = f"{basepath}/{scenario}/{scenario}_{id}.hepmc"
        prop     = event_processor_sequential_topology(filename=filename, prop=prop,
                                              dark_meson_name=dark_meson_name, dark_meson_PDG=dark_meson_PDG)

    OUT_PATH = f'{CWD}/figs/analyzehepmc/{scenario}'
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    
    compute_stats(prop=prop, savename=f'{scenario}', outpath=OUT_PATH)


def pdf_merger(path, savename):
    """
    Merge pdfs
    """    
    pdfiles = []
    for filename in os.listdir(path):
        if filename.endswith('.pdf'):
                if 'MERGED' not in filename:
                        pdfiles.append(filename)
    
    # Sort alphabetically
    pdfiles.sort(key = str.lower)
    
    pdfMerge = PyPDF2.PdfMerger()
    for filename in pdfiles:
        pdfFile = open(f'{path}/{filename}', 'rb')
        pdfReader = PyPDF2.PdfReader(pdfFile)
        pdfMerge.append(pdfReader)
        pdfFile.close()
    pdfMerge.write(f'{path}/MERGED__{savename}.pdf')


def main(basepath):

    # -------------------------
    ## Scenario O. (contact topology)
    
    # Parameter combinations list (add more here)
    param = [{'m': '2', 'ctau': '1',  'xiO': '1', 'xiL': '1', 'file_id': [1]},
             {'m': '2', 'ctau': '50', 'xiO': '1', 'xiL': '1', 'file_id': [1]}]
    
    for p in param:
        
        scenario = f"hiddenValleyGridPack_vector_m_{p['m']}_ctau_{p['ctau']}_xiO_{p['xiO']}_xiL_{p['xiL']}"
        wrapper_contact_topology(p=p, scenario=scenario, basepath=basepath, dark_meson_name="rho3", dark_meson_PDG=PDG_RHO3)
    
    # -------------------------
    ## Scenario A. (sequential topology)

    # Parameter combinations list
    param = [{'mpi': '4',  'mA': '1p33', 'ctau': '10',  'file_id': [1]},
             {'mpi': '10', 'mA': '3p33', 'ctau': '10',  'file_id': [1]},
             {'mpi': '10', 'mA': '3p33', 'ctau': '100', 'file_id': [1]}]
    
    for p in param:
        
        scenario = f"scenarioA_mpi_{p['mpi']}_mA_{p['mA']}_ctau_{p['ctau']}"
        wrapper_sequential_topology(p=p, scenario=scenario, basepath=basepath, dark_meson_name="pi3", dark_meson_PDG=PDG_PI3)
    
    # -------------------------
    ## Scenario B1. (sequential topology)
    
    # Parameter combinations list
    param = [{'mpi': '4', 'mA': '1p33', 'ctau': '10',  'file_id': [1]}]
    
    for p in param:
        
        scenario = f"scenarioB1_mpi_{p['mpi']}_mA_{p['mA']}_ctau_{p['ctau']}_pi3ct"
        wrapper_sequential_topology(p=p, scenario=scenario, basepath=basepath, dark_meson_name="pi3", dark_meson_PDG=PDG_PI3)
    
    # -------------------------
    ## Scenario B2. (sequential topology)
    
    # Parameter combinations list
    param = [{'mpi': '4', 'mA': '2p10', 'ctau': '10',  'file_id': [1]}]
    
    for p in param:
        
        scenario = f"scenarioB2_mpi_{p['mpi']}_mA_{p['mA']}_ctau_{p['ctau']}"
        wrapper_sequential_topology(p=p, scenario=scenario, basepath=basepath, dark_meson_name="eta3", dark_meson_PDG=PDG_ETA3)
    
    # -------------------------
    ## Scenario C. (contact topology)
    
    # Parameter combinations list
    param = [{'mpi': '10', 'mA': '8p00', 'ctau': '10',  'file_id': [1]}]
    
    for p in param:
        
        scenario = f"scenarioC_mpi_{p['mpi']}_mA_{p['mA']}_ctau_{p['ctau']}"
        wrapper_contact_topology(p=p, scenario=scenario, basepath=basepath, dark_meson_name="eta3", dark_meson_PDG=PDG_ETA3)


if __name__ == "__main__":
    
    #basepath = os.getcwd()
    basepath = f'/vols/cms/jleonhol/samples'
    
    main(basepath=basepath)

