import glob
import os
import shutil

import torch
from setuptools import setup
from torch.utils import cpp_extension

# kaitian config
kaitian_path = os.path.dirname(os.path.abspath(__file__))
sources = glob.glob(f"{kaitian_path}/src/*.cpp")
include_dirs = [f"{kaitian_path}/include"]
library_dirs = []
libraries = []
support = []
define_macros = []


def cambricon_mlu_support():
    package_name = "torch_mlu"
    try:
        neuware_path = os.getenv("NEUWARE_HOME", "/usr/local/neuware")
        if not os.path.exists(neuware_path):
            raise FileNotFoundError(
                f"Cambricon Neuware SDK is not installed. Search path: {neuware_path}"
            )
        try:
            pkg = __import__(package_name)
        except ImportError:
            raise ImportError(f"{package_name} is not installed.")
        if not torch.mlu.is_available():
            raise RuntimeError("MLU is not available.")
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
                f"{neuware_path}/include",
                f"{torch_mlu_path}/csrc",
            ]
        )
        library_dirs.extend(
            [
                f"{neuware_path}/lib64",
                f"{torch_mlu_path}/csrc/lib",
            ]
        )
        libraries.extend(["torch_mlu"])
        support.append("Cambricon MLU")
        define_macros.append(("SUPPORT_CAMBRICON_MLU", None))
        print("Cambricon MLU support: ok")
    except Exception as e:
        print(f"Cambricon MLU support: failed. Error: {e}")


cambricon_mlu_support()


def format_list(name, items):
    formatted_items = ",\n  ".join(f"'{item}'" for item in items)
    return f"{name} = [\n  {formatted_items}\n]"


print(format_list("sources", sources))
print(format_list("include_dirs", include_dirs))
print(format_list("library_dirs", library_dirs))
print(format_list("libraries", libraries))
print(format_list("define_macros", define_macros))
print(format_list("support devices", support))

module = cpp_extension.CppExtension(
    name="torch_kaitian",
    sources=sources,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=define_macros,
)

setup(
    name="torch-kaitian",
    version="0.0.0+develop",
    description="KaiTian is a Pytorch backend extension, which unified various collective communications libraries.",
    ext_modules=[module],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
