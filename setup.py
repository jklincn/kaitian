import glob
import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils import cpp_extension

# kaitian config
kaitian_path = Path(__file__).resolve().parent
sources = glob.glob(f"{kaitian_path}/src/*.cpp")
include_dirs = [f"{kaitian_path}/include"]
library_dirs = ["/usr/local/lib"]
libraries = ["hiredis", "gloo"]
define_macros = []
runtime_library_dirs = []
extra_compile_args = []
extra_link_args = []


def mlu_support():
    package_name = "torch_mlu"
    neuware_home = os.getenv("NEUWARE_HOME")
    if neuware_home is None:
        raise EnvironmentError("Environment variable NEUWARE_HOME is not set.")
    if not os.path.exists(neuware_home):
        raise FileNotFoundError(
            f"Cambricon Neuware SDK is not installed. Search path: {neuware_home}"
        )
    pkg = __import__(package_name)
    torch_mlu_path = pkg.__path__[0]
    include_dirs.extend(
        [
            f"{neuware_home}/include",
            f"{torch_mlu_path}/csrc",
            # for '#include "cncl_utils.h"' in include/mlu/process_group_cncl.hpp
            f"{torch_mlu_path}/csrc/framework/distributed",
        ]
    )
    library_dirs.extend([f"{neuware_home}/lib64", f"{torch_mlu_path}/csrc/lib"])
    libraries.extend(["torch_mlu"])
    runtime_library_dirs.extend([f"{torch_mlu_path}/csrc/lib"])
    define_macros.append(("KAITIAN_MLU", None))


def cuda_support():
    cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda")
    if not os.path.exists(cuda_home):
        raise FileNotFoundError(
            f"CUDAToolKit is not installed. Search path: {cuda_home}"
        )
    include_dirs.extend(
        [
            # for '#include <nccl.h>', see kaitian/include/cuda/cuda.hpp
            f"{kaitian_path}/include/cuda",
            f"{cuda_home}/include",
        ]
    )
    library_dirs.extend([f"{cuda_home}/lib64"])
    libraries.extend(["cudart", "torch_cuda"])
    define_macros.extend([("KAITIAN_CUDA", None), ("USE_C10D_NCCL", None)])


def setup_extension():
    device_type = os.environ.get("DEVICE")
    ext_modules = []
    cmdclass = {}
    entry_points = {}
    if device_type:
        match device_type:
            case "MLU":
                mlu_support()
            case "CUDA":
                cuda_support()
            case _:
                raise EnvironmentError(f"Unsupported DEVICE type: {device_type}")
        ext_modules = [
            cpp_extension.CppExtension(
                name="torch_kaitian._C",
                sources=sources,
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=libraries,
                define_macros=define_macros,
                runtime_library_dirs=runtime_library_dirs,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            )
        ]
        cmdclass = {"build_ext": cpp_extension.BuildExtension}
    else:
        entry_points = {"console_scripts": ["kaitian=torch_kaitian.cli:main"]}

    return ext_modules, cmdclass, entry_points


ext_modules, cmdclass, entry_points = setup_extension()

version_file_path = kaitian_path / "version.txt"
with open(version_file_path, "r") as file:
    version = file.read()

setup(
    name="torch_kaitian",
    version=version,
    description="KaiTian is a Pytorch backend extension that enables distributed data parallel for heterogeneous devices.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="jklincn",
    author_email="jklincn@outlook.com",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    entry_points=entry_points,
)
