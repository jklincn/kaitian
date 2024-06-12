import glob
import os

from setuptools import setup, find_packages
from torch.utils import cpp_extension

# kaitian config
kaitian_path = os.path.dirname(os.path.abspath(__file__))
sources = glob.glob(f"{kaitian_path}/src/*.cpp")
include_dirs = [f"{kaitian_path}/include", f"/opt/openmpi/include"]
library_dirs = [f"/opt/openmpi/lib"]
libraries = ["mpi"]
define_macros = []
runtime_library_dirs = [f"/opt/openmpi/lib"]


def mlu_support():
    package_name = "torch_mlu"
    try:
        neuware_home = os.getenv("NEUWARE_HOME")
        if neuware_home is None:
            raise EnvironmentError("Environment variable NEUWARE_HOME is not set.")
        if not os.path.exists(neuware_home):
            raise FileNotFoundError(
                f"Cambricon Neuware SDK is not installed. Search path: {neuware_home}"
            )
        try:
            pkg = __import__(package_name)
        except ImportError:
            raise ImportError(f"{package_name} is not installed.")
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
    except Exception as e:
        raise RuntimeError(f"Cambricon MLU support: failed. Error: {e}")


def cuda_support():
    try:
        cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda")
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
    except Exception as e:
        print(f"CUDA support: failed. Error: {e}")


DEVICE = os.environ.get("DEVICE", None)
match DEVICE:
    case "MLU":
        mlu_support()
    case "CUDA":
        cuda_support()
    case _:
        raise RuntimeError("DEVICE not set")


def format_list(name, items):
    formatted_items = ",\n  ".join(f"'{item}'" for item in items)
    return f"{name} = [\n  {formatted_items}\n]"


module = cpp_extension.CppExtension(
    name="torch_kaitian._C",
    sources=sources,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=define_macros,
    runtime_library_dirs=runtime_library_dirs,
    # extra_compile_args=["-g", "-O0"],
    # extra_link_args=["-g"],
)

setup(
    name="torch_kaitian",
    version="0.0.0",
    description="KaiTian is a Pytorch backend extension that enables distributed data parallel across various devices.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="jklincn",
    author_email="jklincn@outlook.com",
    packages=find_packages(),
    ext_modules=[module],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
