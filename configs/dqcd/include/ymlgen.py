# Automated YAML-file generation
#
# m.mieskolainen@imperial.ac.uk, 2022

import argparse
import os

def dprint(outputfile, string, mode='a'):
  print(string)
  with open(outputfile, mode) as f:
    f.write(f'{string} \n')

def str2float(x):
  """
  Conversion from '10p1' type str representation to float
  """
  if x == 'null':
    return x
  
  if type(x) is str:
    return float(x.replace('p','.'))
  elif type(x) is float:
    return x
  else:
    raise Exception(f'str2float: Input {x} should be either str or float')

def printer(outputfile, process, path, end_name, filename, xs, force_xs, isMC, maxevents_scale, rp, flush_index=0):
  
  if flush_index == 0:
    dprint(outputfile, '', 'w') # Empty it

  i = flush_index
  for m in rp['m']:
    for ctau in rp['ctau']:
      for xi_pair in rp['xi_pair']:

        # MC signal
        if isMC == 'true' and m != 'null':
          param_name   = f'm_{m}_ctau_{ctau}_xiO_{xi_pair[0]}_xiL_{xi_pair[1]}'
          process_name = f'{process}_{param_name}'
          folder_name  = f'{process_name}'
        
        # MC background
        elif isMC == 'true' and m == 'null':
          process_name = f'{process}'  
          folder_name  = f'{process_name}_{end_name}'
        
        # Data
        else:
          process_name = f'{process}'
          folder_name  = f'{end_name}'

        # Print
        dprint(outputfile, f'# [{i}]')
        dprint(outputfile, f'{path}--{process_name}: &{path}--{process_name}')
        dprint(outputfile, f"  path:  \'{path}/{folder_name}\'")
        dprint(outputfile, f"  files: \'{filename}\'")
        dprint(outputfile, f'  xs:   {xs}')
        dprint(outputfile, f'  model_param:')
        dprint(outputfile, f'    m:    {str2float(m)}')
        dprint(outputfile, f'    ctau: {str2float(ctau)}')
        dprint(outputfile, f'    xiO:  {str2float(xi_pair[0])}')
        dprint(outputfile, f'    xiL:  {str2float(xi_pair[1])}')
        dprint(outputfile, f'  force_xs: {force_xs}')
        dprint(outputfile, f'  isMC:     {isMC}')
        dprint(outputfile, f'  maxevents_scale: {maxevents_scale}')
        dprint(outputfile, f'')

        i += 1


def printer_newmodels(outputfile, process, path, end_name, filename, xs, force_xs, isMC, maxevents_scale, rp, flush_index=0):
  
  if flush_index == 0:
    dprint(outputfile, '', 'w') # Empty it

  i = flush_index
  for mpi_mA_pair in rp['mpi_mA_pair']:
      for ctau in rp['ctau']:

        # MC signal
        if isMC == 'true' and mpi_mA_pair[0] != 'null':
          param_name   = f'mpi_{mpi_mA_pair[0]}_mA_{mpi_mA_pair[1]}_ctau_{ctau}'
          process_name = f'{process}_{param_name}'  
          folder_name  = f'{process_name}'

        # MC background
        elif isMC == 'true' and mpi_mA_pair[0] == 'null':
          process_name = f'{process}'  
          folder_name  = f'{process_name}_{end_name}'
        
        # Data
        else:
          process_name = f'{process}'
          folder_name  = f'{end_name}'

        # Print
        dprint(outputfile, f'# [{i}]')
        dprint(outputfile, f'{path}--{process_name}: &{path}--{process_name}')
        dprint(outputfile, f"  path:  \'{path}/{folder_name}\'")
        dprint(outputfile, f"  files: \'{filename}\'")
        dprint(outputfile, f'  xs:   {xs}')
        dprint(outputfile, f'  model_param:')
        dprint(outputfile, f'    mpi:    {str2float(mpi_mA_pair[0])}')
        dprint(outputfile, f'    mA:     {str2float(mpi_mA_pair[1])}')
        dprint(outputfile, f'    ctau:   {str2float(ctau)}')
        dprint(outputfile, f'  force_xs: {force_xs}')
        dprint(outputfile, f'  isMC:     {isMC}')
        dprint(outputfile, f'  maxevents_scale: {maxevents_scale}')
        dprint(outputfile, f'')

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
  rp['xi_pair']   = [['1', '1'], ['2p5', '1'], ['2p5', '2p5']]

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

  process         = 'hiddenValleyGridPack_vector'

  # ------------------------------------------
  # Basic
  filename        = f'data_{filerange}.root'
  path            = 'bparkProductionAll_V1p3'
  end_name        = ''
  xs              = '1.0 # [pb]'
  force_xs        = 'true'
  isMC            = 'true'
  maxevents_scale = '1.0'
  # ------------------------------------------
  
  rp = {}
  rp['m']       = ['2', '5', '10', '15', '20']
  rp['ctau']    = ['1', '10', '50', '100', '500'] 
  rp['xi_pair'] = [['1', '1']]
  
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
  rp['m']       = ['10', '15', '20']
  rp['ctau']    = ['10', '50', '100', '500'] 
  rp['xi_pair'] = [['1', '1'], ['2p5', '1'], ['2p5', '2p5']]
  
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


def scenarioA(outputfile, filerange='*'):

  process         = 'scenarioA'

  # ------------------------------------------
  # Basic
  filename        = f'data_{filerange}.root'
  path            = 'bparkProductionAll_V1p3'
  end_name        = ''
  xs              = '1.0 # [pb]'
  force_xs        = 'true'
  isMC            = 'true'
  maxevents_scale = '1.0'
  # ------------------------------------------
  
  rp = {}
  rp['mpi_mA_pair']  = [['1', '0p33'],['2', '0p67'],['4','0p40'],['4','1p33'],['10','1p00'],['10','3p33']]
  rp['ctau']         = ['0p1','1p0','10','100']
  
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
  printer_newmodels(**param)

def scenarioB1(outputfile, filerange='*'):

  process         = 'scenarioB1'

  # ------------------------------------------
  # Basic
  filename        = f'data_{filerange}.root'
  path            = 'bparkProductionAll_V1p3'
  end_name        = ''
  xs              = '1.0 # [pb]'
  force_xs        = 'true'
  isMC            = 'true'
  maxevents_scale = '1.0'
  # ------------------------------------------
  
  rp = {}
  rp['mpi_mA_pair']  = [['1', '0p33'],['2', '0p67'],['2','0p40'],['4','0p80'],['4','1p33']]
  rp['ctau']         = ['0p1','1p0','10','100']
  
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
  printer_newmodels(**param)


def scenarioB2(outputfile, filerange='*'):

  process         = 'scenarioB2'

  # ------------------------------------------
  # Basic
  filename        = f'data_{filerange}.root'
  path            = 'bparkProductionAll_V1p3'
  end_name        = ''
  xs              = '1.0 # [pb]'
  force_xs        = 'true'
  isMC            = 'true'
  maxevents_scale = '1.0'
  # ------------------------------------------
  
  rp = {}
  rp['mpi_mA_pair']  = [['1', '0p60'],['2', '1p10'],['4','2p10']]
  rp['ctau']         = ['0p1','1p0','10','100']
  
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
  printer_newmodels(**param)


def scenarioC(outputfile, filerange='*'):

  process         = 'scenarioC'

  # ------------------------------------------
  # Basic
  filename        = f'data_{filerange}.root'
  path            = 'bparkProductionAll_V1p3'
  end_name        = ''
  xs              = '1.0 # [pb]'
  force_xs        = 'true'
  isMC            = 'true'
  maxevents_scale = '1.0'
  # ------------------------------------------
  
  rp = {}
  rp['mpi_mA_pair']  = [['2', '1p60'],['4', '3p20'],['10','8p00']]
  rp['ctau']         = ['0p1','1p0','10','100']
  
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
  printer_newmodels(**param)


def QCD(outputfile, filerange='*', paramera='old'):

  processes = [ \
  {'path':     'bparkProductionAll_V1p3',
   'process':  'QCD_Pt-15To20_MuEnrichedPt5_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2',
   'end_name': 'MINIAODSIM_v1p1_generationSync',
   'xs': 2799000.0} # pb
  ,
  
  {'path':     'bparkProductionAll_V1p3',
   'process':  'QCD_Pt-20To30_MuEnrichedPt5_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2',
   'end_name': 'MINIAODSIM_v1p1_generationSync',
   'xs': 2526000.0 }
  ,

  {'path':     'bparkProductionAll_V1p3',
   'process':  'QCD_Pt-30To50_MuEnrichedPt5_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2',
   'end_name': 'MINIAODSIM_v1p1_generationSync',
   'xs': 1362000.0}
  ,

  {'path':     'bparkProductionAll_V1p3',
   'process':  'QCD_Pt-50To80_MuEnrichedPt5_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2',
   'end_name': 'MINIAODSIM_v1p1_generationSync',
   'xs': 376600.0}
  ,

  {'path':     'bparkProductionAll_V1p3',
   'process':  'QCD_Pt-80To120_MuEnrichedPt5_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2',
   'end_name': 'MINIAODSIM_v1p1_generationSync',
   'xs': 88930.0} 
  ,

  {'path':     'bparkProductionAll_V1p3',
   'process':  'QCD_Pt-120To170_MuEnrichedPt5_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2',
   'end_name': 'MINIAODSIM_v1p1_generationSync',
   'xs': 21230.0}  
  ,

  {'path':     'bparkProductionAll_V1p3',
   'process':  'QCD_Pt-170To300_MuEnrichedPt5_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2',
   'end_name': 'MINIAODSIM_v1p1_generationSync',
   'xs': 7055.0}
  ,
  {'path':     'bparkProductionAll_V1p3',
   'process':  'QCD_Pt-300To470_MuEnrichedPt5_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2',
   'end_name': 'MINIAODSIM_v1p1_generationSync',
   'xs': 619.3} 
  ,

  {'path':     'bparkProductionAll_V1p3',
   'process':  'QCD_Pt-470To600_MuEnrichedPt5_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2',
   'end_name': 'MINIAODSIM_v1p1_generationSync',
   'xs': 59.24}
  ,

  {'path':     'bparkProductionAll_V1p3',
   'process':  'QCD_Pt-600To800_MuEnrichedPt5_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2',
   'end_name': 'MINIAODSIM_v1p1_generationSync',
   'xs': 18.21}
  ,

  {'path':     'bparkProductionAll_V1p3',
   'process':  'QCD_Pt-800To1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2',
   'end_name': 'MINIAODSIM_v1p1_generationSync',
   'xs': 3.275}
  ,

  {'path':     'bparkProductionAll_V1p3',
   'process':  'QCD_Pt-1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2',
   'end_name': 'MINIAODSIM_v1p1_generationSync',
   'xs': 1.078} ]
  
  rp = {}

  if   paramera == 'old':
      
    rp['m']       = ['null']
    rp['ctau']    = ['null'] 
    rp['xi_pair'] = [['null', 'null']]
    rp['xi2str']  = ['null']
    
    pfunc = printer
    
  elif paramera == 'new':
    
    rp['mpi_mA_pair'] = [['null', 'null']]
    rp['ctau']        = ['null']

    pfunc = printer_newmodels
  
  else:
    raise Exception('Unknown model string')

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
      pfunc(**param)
    else:
      pfunc(**param, flush_index=i)

     
def data(outputfile, filerange='*', period='B', paramera='old'):

  processes = None
  
  if   period == 'B':
    
    processes = [
    {'path':     'bparkProductionAll_V1p0',
     'process':  'ParkingBPH1_Run2018B',
     'end_name': 'ParkingBPH1_Run2018B-05May2019-v2_MINIAOD_v1p0_generationSync'
    }
    ]

  elif period == 'C':
    
    processes = [
    {'path':     'bparkProductionAll_V1p0',
     'process':  'ParkingBPH1_Run2018C',
     'end_name': 'ParkingBPH1_Run2018C-05May2019-v1_MINIAOD_v1p0_generationSync'
    }
    ]
  
  elif period == 'D':
    
    processes = [
    {'path':     'bparkProductionAll_V1p3',
     'process':  'ParkingBPH1_Run2018D-UL2018_MiniAODv2-v1',
     'end_name': 'ParkingBPH1_Run2018D-UL2018_MiniAODv2-v1_MINIAOD_v1p3_generationSync'
    }
    ]
  
  else:
    raise Exception(__name__ + f'.data: Unknown period "{period}" chosen')
  
  rp = {}
  
  if   paramera == 'old':
    
    rp['m']         = ['null']
    rp['ctau']      = ['null'] 
    rp['xi_pair']   = [['null', 'null']]
    rp['xi2str']    = ['null']
    
    pfunc = printer
    
  elif paramera == 'new':
    
    rp['mpi_mA_pair'] = [['null', 'null']]
    rp['ctau']    = ['null']
    
    pfunc = printer_newmodels
      
  else:
    raise Exception('Unknown model type')
  
  for i in range(len(processes)):

    # ------------------------------------------
    # Basic
    filename        = f'output_{filerange}.root'
    force_xs        = 'false'
    isMC            = 'false'
    xs              =  1.0
    maxevents_scale = '1.0'
    # ------------------------------------------

    param = {
      'outputfile':      outputfile,
      'rp':              rp,
      'process':         processes[i]['process'],
      'path':            processes[i]['path'],
      'end_name':        processes[i]['end_name'],
      'filename':        filename,
      'xs':              xs,
      'force_xs':        force_xs,
      'isMC':            isMC,
      'maxevents_scale': maxevents_scale
    }

    if i == 0:
      pfunc(**param)
    else:
      pfunc(**param, flush_index=i)


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Generate some YAML-files.')
  parser.add_argument('--process',    type=str, default='vector')
  parser.add_argument('--paramera',   type=str, default='new')
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
  
  elif args.process == 'scenarioA':
    scenarioA(outputfile=outputfile, filerange=args.filerange)

  elif args.process == 'scenarioB1':
    scenarioB1(outputfile=outputfile, filerange=args.filerange)

  elif args.process == 'scenarioB2':
    scenarioB2(outputfile=outputfile, filerange=args.filerange)

  elif args.process == 'scenarioC':
    scenarioC(outputfile=outputfile, filerange=args.filerange)

  elif args.process == 'QCD':
    QCD(outputfile=outputfile, filerange=args.filerange,  paramera=args.paramera)
  
  elif args.process == 'data-B':
    data(outputfile=outputfile, filerange=args.filerange, period='B', paramera=args.paramera)

  elif args.process == 'data-C':
    data(outputfile=outputfile, filerange=args.filerange, period='C', paramera=args.paramera)
  
  elif args.process == 'data-D':
    data(outputfile=outputfile, filerange=args.filerange, period='D', paramera=args.paramera)
  
  else:
    print('Error: unknown --process chosen (run --help)')

  print(f'Saved to file "{outputfile}"')
