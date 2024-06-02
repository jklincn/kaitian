#include "support/cambricon_mlu.hpp"

#include "device.hpp"
#include "scheduler.hpp"
#include "util.hpp"

void find_mlu() {
    unsigned int count;
    cnrtGetDeviceCount(&count);
    cndevCheckErrors(cndevInit(0));

    for (unsigned int i = 0; i < count; i++) {
        cnrtDeviceProp_t prop;
        cnrtGetDeviceProperties(&prop, i);
        std::string bdf =
            get_pcie_bdf(prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
        // Check the health status of the device
        cndevCardHealthState_t health;
        health.version = CNDEV_VERSION_5;
        cndevCheckErrors(cndevGetCardHealthState(&health, i));
        if (health.health == 0 ||
            health.deviceState == CNDEV_HEALTH_STATE_DEVICE_IN_PROBLEM) {
            std::cout << "error: Cambricon MLU " << i
                      << " health state in problem" << prop.name << std::endl;
            continue;
        }
        kaitian::Device mlu_device(
            "Cambricon " + std::string(prop.name), bdf,
            kaitian::DeviceType::MLU,
            (unsigned long)prop.availableGlobalMemorySize * 1024 * 1024,
            c10::Device(c10::DeviceType::MLU, i));
        scheduler.add_device(mlu_device);
    }
}