# A tool to verify HepMC(2) event content properties
#
# For DQCD scenarios: https://arxiv.org/pdf/2303.04167.pdf
#
# Run with: python icemc/analyzehepmc.py 
# 
# m.mieskolainen@imperial.ac.uk, 2023

import matplotlib.pyplot as plt
import pyhepmc
import numpy as np
import os
import PyPDF2

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

def event_processor_2_to_4(filename, prop, eprint: int=100, print_first=True):
    """
    Process event content
    """
    with pyhepmc.open(filename) as f:
        ev = 0
        for event in f:
            if ev == 0 and print_first:
                print(event)
                print("")
            
            for v in event.vertices:
                
                # pi3 -> A'A' vertex
                if any(q.pid == 4900111 for q in v.particles_in) and any(q.pid == 999999 for q in v.particles_out):
                    
                    # Vertex and pi3 momentum
                    vertex = v.position
                    mom    = v.particles_in[0].momentum
                    
                    dxy  = np.sqrt(vertex.x**2 + vertex.y**2)
                    d3   = np.sqrt(vertex.x**2 + vertex.y**2 + vertex.z**2)
                    
                    # decay length in lab = c*tau*beta*gamma, where tau the lifetime in the rest frame
                    ctau = d3 / (beta(mom) * gamma(mom))
                    
                    prop["vertex(pi3->A'A')"]["dxy"].append(dxy)
                    prop["vertex(pi3->A'A')"]["d3"].append(d3)
                    prop["vertex(pi3->A'A')"]["ctau"].append(ctau)
                    
                    pt  = mom.pt()
                    eta = mom.eta()
                    m   = mom.m()
                    
                    prop["momentum(pi3)"]["pt"].append(pt)
                    prop["momentum(pi3)"]["eta"].append(eta)
                    prop["momentum(pi3)"]["m"].append(m)
                    
                    if eprint is not None and ev % eprint == 0:
                        print(f"pi3 -> A'A' vertex (event = {ev}) [{filename}]")
                        print(f"vertex: {v}")
                        print(f"in:     {v.particles_in}")
                        print(f"out:    {v.particles_out}")
                        print("")
                
                # A' -> mu+mu-
                if any(q.pid == 999999 for q in v.particles_in) and any(abs(q.pid) == 13 for q in v.particles_out):
                    
                    # Vertex and A' momentum
                    vertex = v.position
                    mom    = v.particles_in[0].momentum
                    
                    dxy  = np.sqrt(vertex.x**2 + vertex.y**2)
                    d3   = np.sqrt(vertex.x**2 + vertex.y**2 + vertex.z**2)
                    ctau = d3 / (beta(mom) * gamma(mom))
                    
                    prop["vertex(A'->mu+mu-)"]["dxy"].append(dxy)
                    prop["vertex(A'->mu+mu-)"]["d3"].append(d3)
                    prop["vertex(A'->mu+mu-)"]["ctau"].append(ctau)
                    
                    pt  = mom.pt()
                    eta = mom.eta()
                    m   = mom.m()
                    
                    prop["momentum(A')"]["pt"].append(pt)
                    prop["momentum(A')"]["eta"].append(eta)
                    prop["momentum(A')"]["m"].append(m)
                    
                    # Muon momentum, pick the first [0]
                    mom = v.particles_out[0].momentum
                    pt  = mom.pt()
                    eta = mom.eta()
                    
                    prop["momentum(mu+)"]["pt"].append(pt)
                    prop["momentum(mu+)"]["eta"].append(eta)
                    
                    if eprint is not None and ev % eprint == 0:
                        print(f"A' -> mu+mu- vertex (event = {ev}) [{filename}]")
                        print(f"vertex: {v}")
                        print(f"in:     {v.particles_in}")
                        print(f"out:    {v.particles_out}")
                        print("")
                    
            ev += 1
    
    return prop


def compute_stats(prop, savename, outpath):
    """
    Compute histograms and point statistics
    """
    
    print(f"Sample statistics: {savename}")

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
                plt.hist(x, 100, label=f'$\\mu = {np.mean(x):0.2e}, \\sigma = {np.std(x):0.2e}$')
                
                if i == 0:
                    plt.title(f'{savename} | {key}', color=(0.5, 0, 0))
                    plt.ylabel('counts')
                plt.xlabel(f'{var}')
                
                ax[i].set_box_aspect(1)
                if i == 1:
                    plt.yscale('log')
                    plt.legend(fontsize=7)
            
            filename = f'{outpath}/{savename}_{key}_{var}.pdf'
            plt.savefig(filename, bbox_inches='tight')
            print(f'Saved figure: {filename}')
            plt.close()

    # Merge pdfs
    pdf_merger(path=outpath, savename=savename)


def wrapper_2_to_4(p, scenario, basepath):
    """
    Wrapper function
    """
    
    CWD = os.getcwd()
    
    prop = {
        "vertex(A'->mu+mu-)": {
            "dxy" : [],
            "d3"  : [],
            "ctau": []
        },
        "vertex(pi3->A'A')": {
            "dxy" : [],
            "d3"  : [],
            "ctau": []
        },
        "momentum(pi3)": {
            "pt":  [],
            "eta": [],
            "m":   []
        },
        "momentum(A')": {
            "pt":  [],
            "eta": [],
            "m":   []
        },
        "momentum(mu+)": {
            "pt":  [],
            "eta": []
        }
    }

    # Loop over files
    for id in p['file_id']:
        filename = f"{basepath}/{scenario}/{scenario}_{id}.hepmc"
        prop     = event_processor_2_to_4(filename=filename, prop=prop)

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

    ## 2 -> 4 scenario A.

    # Parameter combinations list (add more here)
    param = [{'mpi': '4', 'mA': '1p33', 'ctau': '10', 'file_id': [1]}]
    
    for p in param:
        
        scenario = f"scenarioA_mpi_{p['mpi']}_mA_{p['mA']}_ctau_{p['ctau']}"
        wrapper_2_to_4(p=p, scenario=scenario, basepath=basepath)
    
    
    ## 2 -> 4 scenario B1.
    
    # Parameter combinations list
    param = [{'mpi': '4', 'mA': '1p33', 'ctau': '10', 'file_id': [1]}]
    
    for p in param:
        
        scenario = f"scenarioB1_mpi_{p['mpi']}_mA_{p['mA']}_ctau_{p['ctau']}"
        wrapper_2_to_4(p=p, scenario=scenario, basepath=basepath)


if __name__ == "__main__":
    
    #basepath = os.getcwd()
    basepath = f'/vols/cms/jleonhol/samples'
    
    main(basepath=basepath)

