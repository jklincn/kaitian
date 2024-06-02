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

#include <string>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

at::Tensor kaitian__copy_from(const at::Tensor& self, const at::Tensor& dst,
                              bool non_blocking);

at::Tensor kaitian_empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                                 c10::optional<at::ScalarType> dtype,
                                 c10::optional<at::Layout> layout,
                                 c10::optional<at::Device> device,
                                 c10::optional<bool> pin_memory);

void set_device(c10::DeviceIndex device_index);
c10::Device kaitian_device();

namespace kaitian {

typedef enum { CUDA, MLU } DeviceType;

class Device {
   public:
    Device(const std::string& name, const std::string& bdf,
           const DeviceType& type, unsigned long memory_capacity,
           const c10::Device& device)
        : _name(name),
          _bdf(bdf),
          _type(type),
          _memory_capacity(memory_capacity),
          _device(device) {};
    std::string name() { return _name; }
    std::string bdf() { return _bdf; }
    c10::Device device() { return _device; }
    unsigned long memory_capacity() { return _memory_capacity; }

   private:
    std::string _name;
    std::string _bdf;
    DeviceType _type;
    unsigned long _memory_capacity;
    c10::Device _device;
};
}  // namespace kaitian
