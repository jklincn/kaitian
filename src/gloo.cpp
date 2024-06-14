#include <gloo/allreduce.h>
#include <gloo/allreduce_ring.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/transport/tcp/device.h>

#include <iostream>
#include <vector>

std::shared_ptr<gloo::rendezvous::Context> context;
int rank;
int world_size;
void mysum(void *c_, const void *a_, const void *b_, int n) {
    printf("n=%d\r\n", n);
    int *c = static_cast<int *>(c_);
    const int *a = static_cast<const int *>(a_);
    const int *b = static_cast<const int *>(b_);
    for (auto i = 0; i < n; i++) {
        printf("a[%d]=%d\r\n", i, a[i]);
        printf("b[%d]=%d\r\n", i, b[i]);
        c[i] = a[i] + b[i];
        printf("c[%d]=%d\r\n", i, c[i]);
    }
}

void gloo_init(const std::string &hostname) {
    auto dev = gloo::transport::tcp::CreateDevice(hostname.c_str());
    auto fileStore = gloo::rendezvous::FileStore("/tmp/gloo");
    rank = atoi(getenv("KAITIAN_RANK"));
    world_size = atoi(getenv("KAITIAN_WORLD_SIZE"));
    context = std::make_shared<gloo::rendezvous::Context>(rank, world_size);
    context->connectFullMesh(fileStore, dev);
    std::cout << "gloo_init success" << std::endl;
}

void test() {
    size_t elements = 4;
    std::vector<int *> inputPointers;
    std::vector<int *> outputPointers;
    for (size_t i = 0; i < elements; i++) {
        int *value = reinterpret_cast<int *>(malloc(sizeof(int)));
        *value = i * (rank + 1);
        inputPointers.push_back(value);
        int *value1 = reinterpret_cast<int *>(malloc(sizeof(int)));
        *value1 = 0;
        outputPointers.push_back(value1);
    }

    // Configure AllreduceOptions struct
    gloo::AllreduceOptions opts_(context);
    opts_.setInputs(inputPointers, 1);
    opts_.setOutputs(outputPointers, 1);
    opts_.setAlgorithm(gloo::AllreduceOptions::Algorithm::RING);
    void (*fn)(void *, const void *, const void *, int) = &mysum;
    opts_.setReduceFunction(fn);
    gloo::allreduce(opts_);

    // Print the result.
    std::cout << "Output: " << std::endl;
    for (int i = 0; i < outputPointers.size(); i++) {
        std::cout << "data[" << i << "] = " << *outputPointers[i] << std::endl;
    }
}