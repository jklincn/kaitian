#include "device.hpp"

#include <c10/core/DeviceType.h>

#include "support.hpp"

struct KaiTianAllocator final : at::Allocator {
    KaiTianAllocator() = default;
    at::DataPtr allocate(size_t nbytes) const override {
        void* data = c10::alloc_cpu(nbytes);
        return {data, data, &ReportAndDelete,
                at::Device(at::DeviceType::PrivateUse1, 0)};
    }

    static void ReportAndDelete(void* ptr) {
        if (!ptr) {
            return;
        }
        c10::free_cpu(ptr);
    }

    at::DeleterFnPtr raw_deleter() const override { return &ReportAndDelete; }
};

// Register our dummy allocator
static KaiTianAllocator global_custom_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_custom_alloc);

at::Tensor kaitian_empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                                 c10::optional<at::ScalarType> dtype,
                                 c10::optional<at::Layout> layout,
                                 c10::optional<at::Device> device,
                                 c10::optional<bool> pin_memory) {
#ifdef KAITIAN_MLU
    c10::DeviceIndex current_device = torch_mlu::current_device();
    at::Device mlu_device(at::kMLU, current_device);
    c10::optional<at::Device> optional_mlu_device = mlu_device;
    at::Tensor ret = torch_mlu::mlu::empty_strided(
        size, stride, dtype, layout, optional_mlu_device, pin_memory);
    return ret;
#endif
#ifdef KAITIAN_CUDA
    c10::DeviceIndex current_device = c10::cuda::current_device();
    at::Device cuda_device(at::kCUDA, current_device);
    c10::optional<at::Device> optional_cuda_device = cuda_device;
    at::Tensor ret = at::native::empty_strided_cuda(
        size, stride, dtype, layout, optional_cuda_device, pin_memory);
    return ret;
#endif
    throw std::runtime_error("empty_strided: no available devices");
}

namespace at {
struct KaiTianDeviceGuardImpl final
    : public c10::impl::DeviceGuardImplInterface {
    KaiTianDeviceGuardImpl() {}
    DeviceType type() const override { return kPrivateUse1; }
    Device exchangeDevice(Device) const override {
        return Device(kPrivateUse1, -1);  // no-op
    }
    Device getDevice() const override { return Device(kPrivateUse1, -1); }

    void setDevice(Device device) const override {}
    void uncheckedSetDevice(Device) const noexcept override {}
    Stream getStream(Device) const noexcept override {
        return Stream(Stream::DEFAULT, Device(kPrivateUse1, -1));
    }
    Stream exchangeStream(Stream) const noexcept override {
        return Stream(Stream::DEFAULT, Device(kPrivateUse1, -1));
    }
    DeviceIndex deviceCount() const noexcept override { return 1; }

    void destroyEvent(void* /*event*/, const DeviceIndex /*device_index*/)
        const noexcept override {}

    bool queryStream(const Stream& /*stream*/) const override { return true; }
    void synchronizeStream(const Stream& /*stream*/) const override {}
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, KaiTianDeviceGuardImpl);
}  // namespace at

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty_strided", &kaitian_empty_strided);
}
