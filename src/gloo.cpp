#include "gloo.hpp"

#include <chrono>
#include <iostream>

#define gloo_entry(func, ...) _callFunction(#func, [&]() { func(__VA_ARGS__); })

std::chrono::microseconds total_time(0);
std::map<std::string, std::chrono::microseconds> function_times;

void _callFunction(const std::string &func_name,
                   const std::function<void()> &func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    function_times[func_name] += duration;
    total_time += duration;
}

// void test() {
//     size_t elements = 4;
//     std::vector<int *> inputPointers;
//     std::vector<int *> outputPointers;
//     for (size_t i = 0; i < elements; i++) {
//         int *value = reinterpret_cast<int *>(malloc(sizeof(int)));
//         *value = i * (gloo_rank + 1);
//         inputPointers.push_back(value);
//         int *value1 = reinterpret_cast<int *>(malloc(sizeof(int)));
//         *value1 = 0;
//         outputPointers.push_back(value1);
//     }

//     // Configure AllreduceOptions struct
//     gloo::AllreduceOptions opts_(context);
//     opts_.setInputs(inputPointers, 1);
//     opts_.setOutputs(outputPointers, 1);
//     opts_.setAlgorithm(gloo::AllreduceOptions::Algorithm::RING);
//     void (*fn)(void *, const void *, const void *, int) = &mysum;
//     opts_.setReduceFunction(fn);
//     gloo::allreduce(opts_);

//     // Print the result.
//     std::cout << "Output: " << std::endl;
//     for (int i = 0; i < outputPointers.size(); i++) {
//         std::cout << "data[" << i << "] = " << *outputPointers[i] <<
//         std::endl;
//     }
// }

// https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype
void entry(const std::shared_ptr<gloo::rendezvous::Context> &context,
           torch::Tensor &tensor, GlooFunction op) {
    auto start = std::chrono::high_resolution_clock::now();
    auto origin_device = tensor.device();
    auto tensor_cpu = tensor.to(torch::kCPU);
    // switch (tensor_cpu.scalar_type()) {
    //     case c10::ScalarType::Float:
    //         runBroadcast<float>(context, tensor_cpu);
    //         break;
    //     case c10::ScalarType::Double:
    //         runBroadcast<double>(context, tensor_cpu);
    //         break;
    //     case c10::ScalarType::Long:
    //         runBroadcast<int64_t>(context, tensor_cpu);
    //         break;
    //     case c10::ScalarType::Int:
    //         runBroadcast<int32_t>(context, tensor_cpu);
    //         break;
    //     default:
    //         std::cerr << "Unsupported tensor dtype: "
    //                   << tensor_cpu.scalar_type() << std::endl;
    // }
    tensor = tensor_cpu.to(origin_device);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // function_times[func_name] += duration;
    total_time += duration;
}

void time_spend() {
    double seconds = total_time.count() / 1000000.0;
    std::cout << std::fixed << std::setprecision(3)
              << "Time spent on Kaitian: " << seconds << " seconds"
              << std::endl;

    std::cout << std::left << std::setw(15) << "Function Name" << std::right
              << std::setw(15) << "Time (seconds)" << std::setw(15)
              << "Percentage (%)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    // 输出每个函数的时间和百分比
    for (const auto &pair : function_times) {
        const std::string &funcName = pair.first;
        const std::chrono::microseconds &funcTime = pair.second;
        double percentage = 100.0 * funcTime.count() / total_time.count();
        double funcSeconds = funcTime.count() / 1000000.0;
        std::cout << std::left << std::setw(15) << funcName << std::right
                  << std::setw(15) << std::fixed << std::setprecision(3)
                  << funcSeconds << std::setw(15) << std::fixed
                  << std::setprecision(3) << percentage << std::endl;
    }
}