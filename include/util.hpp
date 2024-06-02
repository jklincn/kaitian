#pragma once

#include <string>

std::string get_pcie_bdf(unsigned int domain, unsigned int bus,
                         unsigned int device, unsigned int function = 0);