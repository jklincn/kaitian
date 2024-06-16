#include "gloo.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>

std::string GlooFunctionToString(GlooFunction op) {
    switch (op) {
        case BROADCAST:
            return "BROADCAST";
        case ALLREDUCE:
            return "ALLREDUCE";
        default:
            return "UNKNOWN";
    }
}

std::chrono::microseconds total_time(0);
std::map<GlooFunction, std::chrono::microseconds> function_times;

// https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype
void gloo_entry(const std::shared_ptr<gloo::rendezvous::Context> &context,
                torch::Tensor &tensor, GlooFunction op) {
    auto start = std::chrono::high_resolution_clock::now();
    auto tensor_cpu = tensor.to(torch::kCPU);
    switch (tensor_cpu.scalar_type()) {
        case c10::ScalarType::Float:
            _entry<float>(context, tensor_cpu, op);
            break;
        case c10::ScalarType::Double:
            _entry<double>(context, tensor_cpu, op);
            break;
        case c10::ScalarType::Long:
            _entry<int64_t>(context, tensor_cpu, op);
            break;
        case c10::ScalarType::Int:
            _entry<int32_t>(context, tensor_cpu, op);
            break;
        default:
            std::cerr << "Unsupported tensor dtype: "
                      << tensor_cpu.scalar_type() << std::endl;
    }
    tensor.copy_(tensor_cpu.to(tensor.device()));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    function_times[op] += duration;
    total_time += duration;
}

void time_spend() {
    double seconds = total_time.count() / 1000000.0;
    std::cout << std::fixed << std::setprecision(3)
              << "Time spent on Kaitian: " << seconds << " seconds"
              << std::endl;

    // Output the time and percentage of each function
    std::cout << std::left << std::setw(15) << "Function Name" << std::right
              << std::setw(15) << "Time (seconds)" << std::setw(15)
              << "Percentage (%)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    for (const auto &pair : function_times) {
        const std::string &funcName = GlooFunctionToString(pair.first);
        const std::chrono::microseconds &funcTime = pair.second;
        double percentage = 100.0 * funcTime.count() / total_time.count();
        double funcSeconds = funcTime.count() / 1000000.0;
        std::cout << std::left << std::setw(15) << funcName << std::right
                  << std::setw(15) << std::fixed << std::setprecision(3)
                  << funcSeconds << std::setw(15) << std::fixed
                  << std::setprecision(3) << percentage << std::endl;
    }
}