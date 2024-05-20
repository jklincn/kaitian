#pragma once
#include <c10/core/Device.h>

#include <vector>

#include "device.hpp"

class Scheduler {
   public:
    Scheduler() {}
    void add_device(kaitian::Device device) {
        _device_available.push_back(device);
    }
    unsigned int world_size() { return _device_available.size(); }
    std::vector<kaitian::Device> device_available() {
        return _device_available;
    }

   private:
    std::vector<kaitian::Device> _device_available;
};

extern Scheduler scheduler;

void find_device();
