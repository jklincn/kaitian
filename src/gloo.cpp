#include "gloo.hpp"

#include <chrono>
#include <iostream>
#include <vector>

#define gloo_entry(func, ...) _callFunction(#func, [&]() { func(__VA_ARGS__); })

std::shared_ptr<gloo::rendezvous::Context> context;
int local_rank = -1;
int kaitian_rank = -1;
int kaitian_world_size = -1;
std::string device_type;

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

void _gloo_init(const std::string &hostname, const int &rank) {
    device_type = hostname;
    local_rank = rank;
    auto dev = gloo::transport::tcp::CreateDevice(hostname.c_str());
    auto fileStore = gloo::rendezvous::FileStore("/tmp/gloo");
    kaitian_rank = atoi(getenv("KAITIAN_RANK"));
    kaitian_world_size = atoi(getenv("KAITIAN_WORLD_SIZE"));
    context = std::make_shared<gloo::rendezvous::Context>(kaitian_rank,
                                                          kaitian_world_size);
    context->connectFullMesh(fileStore, dev);
    std::cout << "\033[1;92mKaitian connection established successfully.\033[0m"
              << std::endl;
}

void gloo_init(const std::string &hostname, const int &local_rank) {
    gloo_entry(_gloo_init, hostname, local_rank);
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
void kaitian_broadcast(torch::Tensor &tensor) {
    auto tensor_cpu = tensor.to(torch::kCPU);
    // std::cout << device_type << " old: " << tensor_cpu << std::endl;
    switch (tensor_cpu.scalar_type()) {
        case c10::ScalarType::Float:
            runBroadcast<float>(tensor_cpu);
            break;
        case c10::ScalarType::Long:
            runBroadcast<int64_t>(tensor_cpu);
            break;
        case c10::ScalarType::Int:
            runBroadcast<int32_t>(tensor_cpu);
            break;
        default:
            std::cerr << "Unsupported tensor dtype: "
                      << tensor_cpu.scalar_type() << std::endl;
    }
    // std::cout << device_type << " new: " << tensor_cpu << std::endl;
    // switch (tensors[0].scalar_type()) {
    //     case c10::ScalarType::Byte:
    //     case c10::ScalarType::Char:
    //     case c10::ScalarType::Short:
    //     case c10::ScalarType::Int:
    //     case c10::ScalarType::Long:
    //     case c10::ScalarType::Half:
    //     case c10::ScalarType::Float:
    //     case c10::ScalarType::Double:
    //     case c10::ScalarType::ComplexHalf:
    //     case c10::ScalarType::ComplexFloat:
    //     case c10::ScalarType::ComplexDouble:
    //     case c10::ScalarType::Bool:
    //     case c10::ScalarType::QInt8:
    //     case c10::ScalarType::QUInt8:
    //     case c10::ScalarType::QInt32:
    //     case c10::ScalarType::BFloat16:
    //     case c10::ScalarType::QUInt4x2:
    //     case c10::ScalarType::QUInt2x4:
    //     case c10::ScalarType::Undefined:
    //     case c10::ScalarType::NumOptions:
    //         break;
    //     default:
    //         std::cerr << "Unsupported tensor dtype: "
    //                   << tensors[0].scalar_type() << std::endl;
    // }
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