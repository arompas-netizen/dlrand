"""Builds the C engines into a shared library that ctypes loads (it is not a CPython extension module)."""
import glob
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildSharedLib(build_ext):
    def get_export_symbols(self, ext):
        # No PyInit_* symbol: on Windows the exports come from __declspec(dllexport) (RL_API).
        return []


posix = sys.platform != "win32"
ext = Extension(
    "dlrand._dlrand",
    sources=sorted(glob.glob("src/*.c")),
    include_dirs=["include", "src"],
    define_macros=[("RL_BUILDING", "1")],
    libraries=["m"] if posix else [],
    extra_compile_args=["-std=c99", "-O2", "-fvisibility=hidden"] if posix else ["/O2"],
)

setup(ext_modules=[ext], cmdclass={"build_ext": BuildSharedLib})
