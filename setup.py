import glob
import os

from setuptools import find_packages, setup
from torch.utils import cpp_extension

# kaitian config
kaitian_path = os.path.dirname(os.path.abspath(__file__))
sources = glob.glob(f"{kaitian_path}/src/*.cpp")
include_dirs = [f"{kaitian_path}/include"]
library_dirs = []
libraries = []
define_macros = []
runtime_library_dirs = []
extra_compile_args = []
extra_link_args = ["/usr/local/lib/libgloo.a"]


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


device_type = os.environ.get("DEVICE", None)
match device_type:
    case "MLU":
        mlu_support()
    case "CUDA":
        cuda_support()
    case _:
        raise EnvironmentError(f"Environment variable DEVICE error: {device_type}.")


module = cpp_extension.CppExtension(
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

setup(
    name="torch_kaitian",
    version="0.0.0",
    description="KaiTian is a Pytorch backend extension that enables distributed data parallel for heterogeneous devices.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="jklincn",
    author_email="jklincn@outlook.com",
    packages=find_packages(),
    ext_modules=[module],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
