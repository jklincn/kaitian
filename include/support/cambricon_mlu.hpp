#pragma once

#include "aten/generated/MLUFunctions.h"
#include "cndev.h"
#include "cnrt.h"
#include "framework/core/device.h"
#include "framework/core/guard_impl.h"
#include "framework/distributed/process_group_cncl.hpp"

void find_mlu();
