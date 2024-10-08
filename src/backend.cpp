#include "backend.hpp"

#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <pybind11/chrono.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <cstdlib>

#include "gloo.hpp"
#include "support.hpp"

namespace c10d {
WorkKaiTian::WorkKaiTian(at::Device device) : device_(device) {
    future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()),
        std::vector<at::Device>{device});
}
bool WorkKaiTian::isCompleted() {
    throw std::runtime_error("[kaitian] error: not supported isCompleted");
}

bool WorkKaiTian::isSuccess() const {
    throw std::runtime_error("[kaitian] error: not supported isSuccess");
}

bool WorkKaiTian::wait(std::chrono::milliseconds) { return true; }

c10::intrusive_ptr<c10::ivalue::Future> WorkKaiTian::getFuture() {
    return future_;
}

ProcessGroupKaiTian::ProcessGroupKaiTian(
    const c10::intrusive_ptr<c10d::Store>& store, int rank, int size)
    : ProcessGroup(rank, size), store_(store) {
#ifdef KAITIAN_MLU
    cncl_process_group_ = torch_mlu::ProcessGroupCNCL::createProcessGroupCNCL(
        store, rank, size, std::chrono::seconds(60));
#endif
#ifdef KAITIAN_CUDA
    nccl_process_group_ =
        c10::make_intrusive<ProcessGroupNCCL>(store, rank, size);
#endif
    if (rank == 0) {
        gloo::rendezvous::RedisStore redis("kaitian_redis");
        auto dev = gloo::transport::tcp::CreateDevice(getenv("DEVICE"));
        kaitian_gloo_world_size = atoi(getenv("KAITIAN_GLOO_WORLD_SIZE"));
        context = std::make_shared<gloo::rendezvous::Context>(
            atoi(getenv("KAITIAN_GLOO_RANK")), kaitian_gloo_world_size);
        context->connectFullMesh(redis, dev);
        std::cout
            << "\033[1;92mKaitian connection established successfully.\033[0m"
            << std::endl;
    }
}
// NB: allgather only used by verify_params_across_processes(), so temporarily
// skip it.
c10::intrusive_ptr<Work> ProcessGroupKaiTian::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const AllgatherOptions& opts) {
    auto work = c10::make_intrusive<WorkKaiTian>(outputTensors[0][0].device());
#ifdef KAITIAN_MLU
    work->cncl_work_ =
        cncl_process_group_->allgather(outputTensors, inputTensors, opts);
    work->cncl_work_->wait();
    auto result = work->cncl_work_->result();
#endif
#ifdef KAITIAN_CUDA
    work->nccl_work_ =
        nccl_process_group_->allgather(outputTensors, inputTensors, opts);
    work->nccl_work_->wait();
    auto result = work->nccl_work_->result();
#endif
    if (context) {
    }
    work->future_->markCompleted(result);
    return work;
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::allreduce(
    std::vector<at::Tensor>& tensors, const AllreduceOptions& opts) {
    auto work = c10::make_intrusive<WorkKaiTian>(tensors[0].device());
#ifdef KAITIAN_MLU
    work->cncl_work_ = cncl_process_group_->allreduce(tensors, opts);
    work->cncl_work_->wait();
#endif
#ifdef KAITIAN_CUDA
    work->nccl_work_ = nccl_process_group_->allreduce(tensors, opts);
    work->nccl_work_->wait();
#endif
    if (context) {
        // NB: Temporarily not considering vector length.
        gloo_entry(context, tensors[0], GlooFunction::ALLREDUCE);

        // Here we do division in advance because the world_size in pytorch is
        // incorrect. And then pytorch will do division again.
        if (tensors[0].scalar_type() == c10::ScalarType::Int ||
            tensors[0].scalar_type() == c10::ScalarType::Long) {
            auto t = tensors[0].to(torch::kFloat32);
            t.div_(kaitian_gloo_world_size);
            tensors[0].copy_(t.round().to(tensors[0].scalar_type()));
        } else {
            tensors[0].div_(kaitian_gloo_world_size);
        }
    }
#ifdef KAITIAN_MLU
    work->cncl_work_ = cncl_process_group_->broadcast(tensors);
    work->cncl_work_->wait();
    auto result = work->cncl_work_->result();
#endif
#ifdef KAITIAN_CUDA
    work->nccl_work_ = nccl_process_group_->broadcast(tensors);
    work->nccl_work_->wait();
    auto result = work->nccl_work_->result();
#endif
    work->future_->markCompleted(result);
    return work;
}

// NB: Currently broadcast don't use future.
c10::intrusive_ptr<Work> ProcessGroupKaiTian::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
    auto work = c10::make_intrusive<WorkKaiTian>(tensors[0].device());
#ifdef KAITIAN_MLU
    work->cncl_work_ = cncl_process_group_->broadcast(tensors, opts);
    work->cncl_work_->wait();
#endif
#ifdef KAITIAN_CUDA
    work->nccl_work_ = nccl_process_group_->broadcast(tensors, opts);
    work->nccl_work_->wait();
#endif
    if (context) {
        // NB: Temporarily not considering vector length.
        gloo_entry(context, tensors[0], GlooFunction::BROADCAST);
    }
#ifdef KAITIAN_MLU
    work->cncl_work_ = cncl_process_group_->broadcast(tensors);
    work->cncl_work_->wait();
#endif
#ifdef KAITIAN_CUDA
    work->nccl_work_ = nccl_process_group_->broadcast(tensors);
    work->nccl_work_->wait();
#endif
    return work;
}

c10::intrusive_ptr<ProcessGroup> ProcessGroupKaiTian::createProcessGroupKaiTian(
    const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size,
    const std::chrono::duration<float>& /* unused */) {
    return c10::make_intrusive<ProcessGroupKaiTian>(store, rank, size);
}

}  // namespace c10d

PYBIND11_MODULE(_C, m) {
    m.def("createProcessGroupKaiTian",
          &c10d::ProcessGroupKaiTian::createProcessGroupKaiTian);
    m.def("time_spend", &time_spend);
}