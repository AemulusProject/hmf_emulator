from hmf import *
import cffi, glob, os

__author__ = "Tom McClintock <mcclintock@bnl.gov>"

hmf_dir = os.path.dirname(__file__)
include_dir = os.path.join(hmf_dir,'src')
lib_file    = os.path.join(hmf_dir,'_hmf.so')
# Some installation (e.g. Travis with python 3.x)
# name this e.g. _hmf.cpython-34m.so,
# so if the normal name doesn't exist, look for something else.
if not os.path.exists(lib_file):
    alt_files = glob.glob(os.path.join(os.path.dirname(__file__),'_hmf*.so'))
    if len(alt_files) == 0:
        raise IOError("No file '_hmf.so' found in %s"%hmf_dir)
    if len(alt_files) > 1:
        raise IOError("Multiple files '_hmf*.so' found in %s: %s"%(hmf_dir,alt_files))
    lib_file = alt_files[0]

_ffi = cffi.FFI()
for file_name in glob.glob(os.path.join(include_dir,'*.h')):
    _ffi.cdef(open(file_name).read())
_lib = _ffi.dlopen(lib_file)
