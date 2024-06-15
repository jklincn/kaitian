#pragma once
#include <gloo/algorithm.h>
#include <gloo/allgather.h>
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
#include <gloo/rendezvous/store.h>
#include <gloo/scatter.h>
#include <gloo/transport/tcp/device.h>
#include <torch/torch.h>

enum GlooFunction { BROADCAST, ALLREDUCE, ALLGATHER };

void _callFunction(const std::string &func_name,
                   const std::function<void()> &func);

template <typename T>
void runBroadcast(const std::shared_ptr<gloo::rendezvous::Context> &context,
                  torch::Tensor &tensor) {
    static_assert(std::is_same<T, float>::value ||
                      std::is_same<T, int64_t>::value ||
                      std::is_same<T, int32_t>::value,
                  "Unsupported tensor type for gloo broadcast");

    gloo::BroadcastOneToAll<T> algorithm(
        context, {reinterpret_cast<T *>(tensor.data_ptr())}, tensor.numel(), 0,
        0);

    algorithm.run();
}

void entry(const std::shared_ptr<gloo::rendezvous::Context> &context,
           torch::Tensor &tensor, GlooFunction op);

void time_spend();