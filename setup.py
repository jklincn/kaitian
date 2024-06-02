import glob
import os
import shutil

import torch
from setuptools import setup, find_packages
from torch.utils import cpp_extension

# kaitian config
kaitian_path = os.path.dirname(os.path.abspath(__file__))
sources = glob.glob(f"{kaitian_path}/src/*.cpp")
include_dirs = [f"{kaitian_path}/include"]
library_dirs = []
libraries = []
define_macros = []
# runtime_library_dirs = ["$ORIGIN/lib"]


def cambricon_mlu_support():
    package_name = "torch_mlu"
    try:
        neuware_home = os.getenv("NEUWARE_HOME", "/usr/local/neuware")
        if not os.path.exists(neuware_home):
            raise FileNotFoundError(
                f"Cambricon Neuware SDK is not installed. Search path: {neuware_home}"
            )
        try:
            pkg = __import__(package_name)
        except ImportError:
            raise ImportError(f"{package_name} is not installed.")
        if not torch.mlu.is_available():
            raise RuntimeError("MLU is not available")
        if torch.mlu.device_count() == 0:
            raise RuntimeError("No MLU device")
        torch_mlu_path = pkg.__path__[0]
        # because torch_mlu only copy *.h
        shutil.copy(
            f"{kaitian_path}/include/support/process_group_cncl.hpp",
            f"{torch_mlu_path}/csrc/framework/distributed/process_group_cncl.hpp",
        )
        sources.extend(
            [
                f"{kaitian_path}/src/support/cambricon_mlu.cpp",
            ]
        )
        include_dirs.extend(
            [
                f"{neuware_home}/include",
                f"{torch_mlu_path}/csrc",
            ]
        )
        library_dirs.extend(
            [
                f"{neuware_home}/lib64",
                f"{torch_mlu_path}/csrc/lib",
            ]
        )
        libraries.extend(["torch_mlu"])
        define_macros.append(("SUPPORT_CAMBRICON_MLU", None))
        print("Cambricon MLU support: ok")
    except Exception as e:
        print(f"Cambricon MLU support: failed. Error: {e}")


def cuda_support():
    try:
        cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda")
        if not os.path.exists(cuda_home):
            raise FileNotFoundError(f"CUDA is not installed. Search path: {cuda_home}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if torch.mlu.device_count() == 0:
            raise RuntimeError("No Nvidia GPU")
        sources.extend(
            [
                f"{kaitian_path}/src/support/cuda.cpp",
            ]
        )
        include_dirs.extend(
            [
                f"{cuda_home}/include",
            ]
        )
        library_dirs.extend(
            [
                f"{cuda_home}/lib64",
            ]
        )
        libraries.extend(["cudart"])
        define_macros.append(("SUPPORT_CUDA", None))
        print("CUDA support: ok")
    except Exception as e:
        print(f"CUDA support: failed. Error: {e}")


cambricon_mlu_support()
# cuda_support()


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
    # runtime_library_dirs=runtime_library_dirs,
)

setup(
    name="torch_kaitian",
    version="0.0.0",
    description="KaiTian is a Pytorch backend extension, which unified various collective communications libraries.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="jklincn",
    author_email="jklincn@outlook.com",
    packages=find_packages(),
    # package_data={"torch_kaitian": ["lib/*.so*"]},
    ext_modules=[module],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
