#include "device.hpp"

#include "scheduler.hpp"
#include "support/cambricon_mlu.hpp"

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

at::Tensor kaitian__copy_from(const at::Tensor& self, const at::Tensor& dst,
                              bool non_blocking) {
    std::cout << "_copy_from: self device: " << self.device()
              << ", dst device: " << dst.device() << std::endl;
#ifdef SUPPORT_CAMBRICON_MLU
    at::Tensor t = self.toBackend(c10::Backend::MLU);
    return t;
#endif
    throw std::runtime_error("_copy_from: no available devices");
}

at::Tensor kaitian_empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                                 c10::optional<at::ScalarType> dtype,
                                 c10::optional<at::Layout> layout,
                                 c10::optional<at::Device> device,
                                 c10::optional<bool> pin_memory) {
#ifdef SUPPORT_CAMBRICON_MLU
    c10::DeviceIndex current_device = torch_mlu::current_device();
    at::Device mlu_device(at::kMLU, current_device);
    c10::optional<at::Device> optional_mlu_device = mlu_device;
    at::Tensor re = torch_mlu::mlu::empty_strided(
        size, stride, dtype, layout, optional_mlu_device, pin_memory);
    return re;
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
    void uncheckedSetDevice(Device) const noexcept override {
        // no-op
    }
    Stream getStream(Device) const noexcept override {
        // no-op
        return Stream(Stream::DEFAULT, Device(kPrivateUse1, -1));
    }
    // NB: These do NOT set the current device
    Stream exchangeStream(Stream) const noexcept override {
        // no-op
        return Stream(Stream::DEFAULT, Device(kPrivateUse1, -1));
    }
    DeviceIndex deviceCount() const noexcept override { return 1; }

    // Event-related functions
    void record(void** /*event*/, const Stream& /*stream*/,
                const DeviceIndex /*device_index*/,
                const EventFlag /*flag*/) const override {
        TORCH_CHECK(false, kPrivateUse1, " backend doesn't support events.");
    }
    void block(void* /*event*/, const Stream& /*stream*/) const override {
        TORCH_CHECK(false, kPrivateUse1, " backend doesn't support events.")
    }
    bool queryEvent(void* /*event*/) const override {
        TORCH_CHECK(false, kPrivateUse1, " backend doesn't support events.")
    }
    void destroyEvent(void* /*event*/, const DeviceIndex /*device_index*/)
        const noexcept override {}

    // Stream-related functions
    bool queryStream(const Stream& /*stream*/) const override { return true; }
    void synchronizeStream(const Stream& /*stream*/) const override {
        // Don't wait for anything.
    }
#ifdef SUPPORT_CAMBRICON_MLU
    torch_mlu::mlu::MLUGuardImpl mlu_guard;
#endif
};
C10_REGISTER_GUARD_IMPL(PrivateUse1, KaiTianDeviceGuardImpl);
}  // namespace at

void set_device(c10::DeviceIndex device_index) {
    std::cout << "[kaitian] set_device: " << static_cast<int>(device_index)
              << std::endl;
#ifdef SUPPORT_CAMBRICON_MLU
    torch_mlu::setDevice(device_index);
#endif
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_copy_from", &kaitian__copy_from);
    m.impl("empty_strided", &kaitian_empty_strided);
}

c10::Device kaitian_device() {
    return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

void init_device(pybind11::module& m) {
    m.def("device", &kaitian_device);
    m.def("set_device", &set_device);
}