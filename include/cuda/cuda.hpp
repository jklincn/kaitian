#pragma once

#include <ATen/cuda/PeerToPeerAccess.h>
#include <c10/cuda/CUDAFunctions.h>

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

// used by torch/csrc/distributed/c10d/NCCLUtils.hpp:11:10
#include "nccl.h"