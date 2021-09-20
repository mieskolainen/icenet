# Make local packages visible in the syspath

import sys
import os

paths = ['',
		 '/iceplot/',
		 '/icenet/',
		 '/icebrk/',
		 '/iceid/',
		 '/icefit/']

for p in paths:
	fullpath = os.path.abspath('.') + p
	print(fullpath)
	sys.path.insert(0,fullpath)
