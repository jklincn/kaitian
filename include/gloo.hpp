#pragma once
#include <gloo/algorithm.h>
#include <gloo/allgather.h>
#include <gloo/allgather_ring.h>
#include <gloo/allgatherv.h>
#include <gloo/allreduce.h>
#include <gloo/allreduce_ring.h>
#include <gloo/alltoall.h>
#include <gloo/alltoallv.h>
#include <gloo/barrier.h>
#include <gloo/broadcast.h>
#include <gloo/broadcast_one_to_all.h>
#include <gloo/common/error.h>
#include <gloo/config.h>
#include <gloo/context.h>
#include <gloo/gather.h>
#include <gloo/reduce.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/rendezvous/redis_store.h>
#include <gloo/rendezvous/store.h>
#include <gloo/scatter.h>
#include <gloo/transport/tcp/device.h>
#include <torch/torch.h>

#include <stdexcept>

enum GlooFunction { BROADCAST, ALLREDUCE };

void gloo_entry(const std::shared_ptr<gloo::rendezvous::Context> &context,
                torch::Tensor &tensor, GlooFunction op);
void time_spend();

template <typename T>
void _entry(const std::shared_ptr<gloo::rendezvous::Context> &context,
            torch::Tensor &tensor, GlooFunction op) {
    std::unique_ptr<gloo::Algorithm> algorithm;
    switch (op) {
        case BROADCAST:
            algorithm = std::make_unique<gloo::BroadcastOneToAll<T>>(
                context,
                std::vector<T *>{reinterpret_cast<T *>(tensor.data_ptr())},
                tensor.numel(), 0, 0);
            break;
        case ALLREDUCE:
            algorithm = std::make_unique<gloo::AllreduceRing<T>>(
                context,
                std::vector<T *>{reinterpret_cast<T *>(tensor.data_ptr())},
                tensor.numel());
            break;
    }
    if (algorithm) {
        algorithm->run();
    } else {
        throw std::runtime_error(
            "[KaiTian] Internal Error: gloo algorithm is nullptr.");
    }
}