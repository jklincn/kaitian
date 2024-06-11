#pragma once
#include <ATen/EmptyTensor.h>
#include <ATen/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/core/Allocator.h>
#include <c10/core/Backend.h>
#include <c10/core/Device.h>
#include <c10/core/impl/alloc_cpu.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

at::Tensor kaitian_empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                                 c10::optional<at::ScalarType> dtype,
                                 c10::optional<at::Layout> layout,
                                 c10::optional<at::Device> device,
                                 c10::optional<bool> pin_memory);
