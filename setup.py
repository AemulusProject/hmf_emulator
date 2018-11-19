import sys, os, glob
from setuptools import setup, Extension
import subprocess

sources = glob.glob(os.path.join('src','*.c'))
headers = glob.glob(os.path.join('src','*.h'))
try:
    cflags = subprocess.check_output(['gsl-config', '--cflags']).split()
    lflags = subprocess.check_output(['gsl-config', '--libs']).split()
except OSError:
    raise Exception("Error: must have GSL installed and gsl-config working")

ext=Extension("hmf_emulator._hmf_emulator",
              sources,
              depends=headers,
              include_dirs=['src'],
              extra_compile_args=[os.path.expandvars(flag) for flag in cflags],
              extra_link_args=[os.path.expandvars(flag) for flag in lflags])

