#include "support/cuda.hpp"

#include "device.hpp"
#include "scheduler.hpp"
#include "util.hpp"

void find_cuda() {
    int count;
    cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaError_t error = cudaGetDeviceProperties(&prop, i);
        if (error != cudaSuccess) {
            std::cout
                << "CUDA error: Failed to get device properties for device "
                << i << ": " << cudaGetErrorString(error) << std::endl;
            continue;
        }
        std::string bdf =
            get_pcie_bdf(prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
        kaitian::Device cuda_device(
            std::string(prop.name), bdf, kaitian::DeviceType::CUDA,
            prop.totalGlobalMem, c10::Device(c10::DeviceType::CUDA, i));
        scheduler.add_device(cuda_device);
    }
}