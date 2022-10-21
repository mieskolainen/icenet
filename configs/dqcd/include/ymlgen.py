# Automated YAML-file generation
#
# m.mieskolainen@imperial.ac.uk, 2022

import argparse
import os

def dprint(string, mode='a'):
  print(string)
  with open(outputfile, mode) as f:
    f.write(f'{string} \n')
 
def printer(outputfile, process, path, end_name, filename, xs, force_xs, isMC, maxevents_scale, rp, flush_index=0):
  
  if flush_index == 0:
    dprint('', 'w') # Empty it

  i = flush_index
  for m in rp['m']:
    for ctau in rp['ctau']:
      for xi_pair in rp['xi_pair']:

        # MC signal
        if isMC == 'true' and m != 'null':
          param_name   = f'm_{m}_ctau_{ctau}_xiO_{rp["xi2str"][xi_pair[0]]}_xiL_{rp["xi2str"][xi_pair[1]]}'
          process_name = f'{process}_{param_name}'  
          folder_name  = f'{process_name}_{end_name}'

        # MC background
        elif isMC == 'true' and m == 'null':
          process_name = f'{process}'  
          folder_name  = f'{process_name}_{end_name}'
        
        # Data
        else:
          process_name = f'{process}'
          folder_name  = f'{end_name}'

        # Print
        dprint(f'# [{i}]')
        dprint(f'{path}--{process_name}: &{path}--{process_name}')
        dprint(f"  path:  \'{path}/{folder_name}\'")
        dprint(f"  files: \'{filename}\'")
        dprint(f'  xs:   {xs}')
        dprint(f'  model_param:')
        dprint(f'    m:    {m}')
        dprint(f'    ctau: {ctau}')
        dprint(f'    xiO:  {xi_pair[0]}')
        dprint(f'    xiL:  {xi_pair[1]}')
        dprint(f'  force_xs: {force_xs}')
        dprint(f'  isMC:     {isMC}')
        dprint(f'  maxevents_scale: {maxevents_scale}')
        dprint(f'')

        i += 1


def darkphoton(outputfile, filerange='*'):

  process         = 'HiddenValley_darkphoton'

  # ------------------------------------------
  # Basic
  filename        = f'output_{filerange}.root'
  path            = 'bparkProductionV3'
  end_name        = 'privateMC_11X_NANOAODSIM_v3_generationForBParking'
  xs              = '1.0 # [pb]'
  force_xs        = 'true'
  isMC            = 'true'
  maxevents_scale = '1.0'
  # ------------------------------------------

  rp = {}
  rp['m']         = ['2', '5', '10', '15']
  rp['ctau']      = ['10', '50', '100', '500'] 
  rp['xi_pair']   = [['1', '1'], ['2.5', '1'], ['2.5', '2.5']]
  rp['xi2str']    = {'1': '1', '2.5': '2p5', '2.5': '2p5'}

  param = {
    'outputfile':      outputfile,
    'rp':              rp,
    'process':         process,
    'path':            path,
    'end_name':        end_name,
    'filename':        filename,
    'xs':              xs,
    'force_xs':        force_xs,
    'isMC':            isMC,
    'maxevents_scale': maxevents_scale
  }
  printer(**param)


def vector(outputfile, filerange='*'):

  process         = 'HiddenValley_vector'

  # ------------------------------------------
  # Basic
  filename        = f'output_{filerange}.root'
  path            = 'bparkProductionV3'
  end_name        = 'privateMC_11X_NANOAODSIM_v3_generationForBParking'
  xs              = '1.0 # [pb]'
  force_xs        = 'true'
  isMC            = 'true'
  maxevents_scale = '1.0'
  # ------------------------------------------

  rp = {}
  rp['m']         = ['2', '5', '10', '15', '20']
  rp['ctau']      = ['10', '50', '100', '500'] 
  rp['xi_pair']   = [['1', '1']]
  rp['xi2str']    = {'1': '1'}

  param = {
    'outputfile':      outputfile,
    'rp':              rp,
    'process':         process,
    'path':            path,
    'end_name':        end_name,
    'filename':        filename,
    'xs':              xs,
    'force_xs':        force_xs,
    'isMC':            isMC,
    'maxevents_scale': maxevents_scale
  }
  printer(**param)


def higgs(outputfile, filerange='*'):

  process         = 'HiddenValley_higgs'

  # ------------------------------------------
  # Basic
  filename        = f'output_{filerange}.root'
  path            = 'bparkProductionV2'
  end_name        = 'privateMC_11X_NANOAODSIM_v2_generationForBParking'
  xs              = '1.0 # [pb]'
  force_xs        = 'true'
  isMC            = 'true'
  maxevents_scale = '1.0'
  # ------------------------------------------
  
  rp = {}
  rp['m']         = ['10', '15', '20']
  rp['ctau']      = ['10', '50', '100', '500'] 
  rp['xi_pair']   = [['1', '1'], ['2.5', '1'], ['2.5', '2.5']]
  rp['xi2str']    = {'1': '1', '2.5': '2p5', '2.5': '2p5'}

  param = {
    'outputfile':      outputfile,
    'rp':              rp,
    'process':         process,
    'path':            path,
    'end_name':        end_name,
    'filename':        filename,
    'xs':              xs,
    'force_xs':        force_xs,
    'isMC':            isMC,
    'maxevents_scale': maxevents_scale
  }
  printer(**param)


def QCD(outputfile, filerange='*'):

  processes = [ \
  {'path':     'bparkProductionV1_bkg',
   'process':  'QCD_Pt-15to20_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v3_MINIAODSIM_v1_generationForBParking',
   'xs': 2799000.0} # pb

  ,
  {'path':     'bparkProductionV1_bkg',
   'process':  'QCD_Pt-20to30_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v4_MINIAODSIM_v1_generationForBParking',
   'xs': 2526000.0 }

  ,
  {'path':     'bparkProductionV1_bkg',
   'process':  'QCD_Pt-30to50_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v3_MINIAODSIM_v1_generationForBParking',
   'xs': 1362000.0}

  ,
  {'path':     'bparkProductionV1_bkg',
   'process':  'QCD_Pt-50to80_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v3_MINIAODSIM_v1_generationForBParking',
   'xs': 376600.0}

  ,
  {'path':     'bparkProductionV1_bkg',
   'process':  'QCD_Pt-80to120_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM_v1_generationForBParking',
   'xs': 44465.0} # 88930.0 / 2 [two paths treated as "separate" processes]
  ,
  {'path':     'bparkProductionV1_bkg_ext', # Extension sample
   'process':  'QCD_Pt-80to120_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15_ext1-v2_MINIAODSIM_v2_generationForBParking',
   'xs': 44465.0} # 88930.0 / 2 [two paths treated as "separate" processes]

  ,
  {'path':     'bparkProductionV1_bkg',
   'process':  'QCD_Pt-120to170_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM_v1_generationForBParking',
   'xs': 10615.0} # 21230.0 / 2 [two paths treated as "separate" processes]
  ,
  {'path':     'bparkProductionV1_bkg_ext', # Extension sample
   'process':  'QCD_Pt-120to170_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15_ext1-v2_MINIAODSIM_v2_generationForBParking',
   'xs': 10615.0} # 21230.0 / 2 [two paths treated as "separate" processes]

  ,
  {'path':     'bparkProductionV1_bkg',
   'process':  'QCD_Pt-170to300_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v3_MINIAODSIM_v1_generationForBParking',
   'xs': 7055.0}

  ,
  {'path':     'bparkProductionV1_bkg',
   'process':  'QCD_Pt-300to470_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v3_MINIAODSIM_v1_generationForBParking',
   'xs': 309.65} # 619.3 / 2 [two paths treated as "separate" processes]
  ,
  {'path':     'bparkProductionV1_bkg_ext', # Extension sample
   'process':  'QCD_Pt-300to470_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15_ext3-v1_MINIAODSIM_v2_generationForBParking',
   'xs': 309.65} # 619.3 / 2 [two paths treated as "separate" processes]

  ,
  {'path':     'bparkProductionV1_bkg',
   'process':  'QCD_Pt-470to600_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM_v1_generationForBParking',
   'xs': 29.62}  # 59.24 / 2 [two paths treated as "separate" processes]
  ,
  {'path':     'bparkProductionV1_bkg_ext', # Extension sample
   'process':  'QCD_Pt-470to600_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15_ext1-v2_MINIAODSIM_v2_generationForBParking',
   'xs': 29.62}  # 59.24 / 2 [two paths treated as "separate" processes]
  ,
  {'path':     'bparkProductionV1_bkg',
   'process':  'QCD_Pt-600to800_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM_v1_generationForBParking',
   'xs': 18.21}
  ,
  {'path':     'bparkProductionV1_bkg',
   'process':  'QCD_Pt-800to1000_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15_ext3-v2_MINIAODSIM_v1_generationForBParking',
   'xs': 1.6375} # 3.275 / 2 [two paths treated as "separate" processes]
  ,
  {'path':     'bparkProductionV1_bkg_ext', # Extension sample
   'process':  'QCD_Pt-800to1000_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15_ext3-v2_MINIAODSIM_v2_generationForBParking',
   'xs': 1.6375} # 3.275 / 2 [two paths treated as "separate" processes]
  ,
  {'path':     'bparkProductionV1_bkg',
   'process':  'QCD_Pt-1000toInf_MuEnrichedPt5',
   'end_name': 'TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM_v1_generationForBParking',
   'xs': 1.078} ]

  rp = {}
  rp['m']       = ['null']
  rp['ctau']    = ['null'] 
  rp['xi_pair'] = [['null', 'null']]
  rp['xi2str']  = ['null']

  for i in range(len(processes)):

    # ------------------------------------------
    # Basic
    filename        = f'output_{filerange}.root'
    force_xs        = 'false'
    isMC            = 'true'
    maxevents_scale = '1.0'
    # ------------------------------------------

    param = {
      'outputfile':      outputfile,
      'rp':              rp,
      'process':         processes[i]['process'],
      'path':            processes[i]['path'],
      'end_name':        processes[i]['end_name'],
      'filename':        filename,
      'xs':              processes[i]['xs'],
      'force_xs':        force_xs,
      'isMC':            isMC,
      'maxevents_scale': maxevents_scale
    }

    if i == 0:
      printer(**param)
    else:
      printer(**param, flush_index=i)


def data(outputfile, filerange='*'):

  process         = 'ParkingData'

  # ------------------------------------------
  # Basic
  filename        = f'output_{filerange}.root'
  path            = 'bparkProductionV3'
  end_name        = 'ParkingBPH1_Run2018B-05May2019-v2_MINIAOD_v1_generationForBParking'
  xs              = 'null'
  force_xs        = 'false'
  isMC            = 'false'
  maxevents_scale = '1.0'
  # ------------------------------------------
  
  rp = {}
  rp['m']         = ['null']
  rp['ctau']      = ['null'] 
  rp['xi_pair']   = [['null', 'null']]
  rp['xi2str']    = ['null']

  param = {
    'outputfile':      outputfile,
    'rp':              rp,
    'process':         process,
    'path':            path,
    'end_name':        end_name,
    'filename':        filename,
    'xs':              xs,
    'force_xs':        force_xs,
    'isMC':            isMC,
    'maxevents_scale': maxevents_scale
  }
  printer(**param)

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Generate some YAML-files.')
  parser.add_argument('--process',    type=str, default='vector')
  parser.add_argument('--filerange',  type=str, default='*')
  parser.add_argument('--outputfile', type=str, default=None)
  args = parser.parse_args()
  
  if args.outputfile is None:
    # By default, we write to the path of this python script
    path = os.path.abspath(os.path.dirname(__file__))
    outputfile = f'{path}/{args.process}.yml'
  else:
    outputfile = args.outputfile

  if   args.process == 'vector':
    vector(outputfile=outputfile, filerange=args.filerange)

  elif args.process == 'higgs':
    higgs(outputfile=outputfile, filerange=args.filerange)

  elif args.process == 'darkphoton':
    darkphoton(outputfile=outputfile, filerange=args.filerange)

  elif args.process == 'QCD':
    QCD(outputfile=outputfile, filerange=args.filerange)
    
  elif args.process == 'data':
    data(outputfile=outputfile, filerange=args.filerange)

  else:
    print('Error: unknown --process chosen (run --help)')

  print(f'Saved to file "{outputfile}"')
