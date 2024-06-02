#include "scheduler.hpp"

#include <pybind11/chrono.h>

#include <iostream>

#include "support/support.hpp"

Scheduler scheduler;

void find_device() {
    std::cout << "[kaitian] info: Start finding devices" << std::endl;

#ifdef SUPPORT_CUDA
    find_cuda();
#endif

#ifdef SUPPORT_CAMBRICON_MLU
    find_mlu();
#endif
    std::vector<kaitian::Device> devices = scheduler.device_available();
    std::cout << "===============================" << std::endl;
    for (auto i = devices.begin(); i != devices.end(); ++i) {
        std::cout << "Find " << i->name() << std::endl;
        std::cout << i->bdf() << std::endl;
        std::cout << "Memory capacity: " << i->memory_capacity() / 1024 / 1024
                  << " MB" << std::endl;
        std::cout << "Register as \"" << i->device() << "\"" << std::endl;
        if (next(i) != devices.end())
            std::cout << "-------------------------------" << std::endl;
    }
    std::cout << "===============================" << std::endl;
    std::cout << "[kaitian] info: Finish finding devices, devices count: "
              << devices.size() << std::endl;
}

unsigned int init_kaitian() {
    find_device();
    unsigned int world_size = scheduler.world_size();
    if (world_size == 0) {
        std::cout << "[kaitian] error: no available devices" << std::endl;
    }
    return world_size;
}

void init_scheduler(pybind11::module& m) { m.def("init", &init_kaitian); }
