"""Builds the C engines into a shared library that ctypes loads (it is not a CPython extension module)."""
import glob
import sys

from setuptools import Extension, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext


class BuildSharedLib(build_ext):
    def get_export_symbols(self, ext):
        # No PyInit_* symbol: on Windows the exports come from __declspec(dllexport) (RL_API).
        return []


class PlatformWheel(bdist_wheel):
    """The library is loaded with ctypes, so it does not depend on the Python version or ABI.
    Tag the wheel py3-none-<platform>: one wheel per operating system / CPU instead of one per Python version."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False          # it contains a compiled library

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


posix = sys.platform != "win32"
ext = Extension(
    "dlrand._dlrand",
    sources=sorted(glob.glob("src/*.c")),
    include_dirs=["include", "src"],
    define_macros=[("RL_BUILDING", "1")],
    libraries=["m"] if posix else [],
    extra_compile_args=["-std=c99", "-O2", "-fvisibility=hidden"] if posix else ["/O2"],
)

setup(ext_modules=[ext], cmdclass={"build_ext": BuildSharedLib, "bdist_wheel": PlatformWheel})
