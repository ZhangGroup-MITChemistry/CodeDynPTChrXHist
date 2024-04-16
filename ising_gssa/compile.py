#import pyximport; pyximport.install()
#import conly2
import sys
import builtins
import os


sys.stdout.write("Stared Compiling" + '\n')
sys.stdout.flush()
from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules = cythonize('sim.pyx'))
sys.stdout.write("Compiled" + '\n')
sys.stdout.flush()
import sim
#conly.run_sim()
