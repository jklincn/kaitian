#include "util.hpp"

#include <iomanip>
#include <sstream>

std::string get_pcie_bdf(unsigned int domain, unsigned int bus,
                         unsigned int device, unsigned int function) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(4) << domain;
    std::string domain_id = oss.str();
    oss.str("");
    oss.clear();
    oss << std::hex << std::setfill('0') << std::setw(2) << bus;
    std::string bus_id = oss.str();
    oss.str("");
    oss.clear();
    oss << std::hex << std::setfill('0') << std::setw(2) << device;
    std::string device_id = oss.str();
    std::string bdf = "BDF: " + domain_id + ":" + bus_id + ":" + device_id +
                      "." + std::to_string(function);
    return bdf;
}

