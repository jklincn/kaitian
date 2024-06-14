#include "backend.hpp"

#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <pybind11/chrono.h>

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
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const AllgatherOptions& opts) {
    auto work = c10::make_intrusive<WorkKaiTian>(outputTensors[0][0].device());
    work->operation_ = OperationType::ALLGATHER;
#ifdef KAITIAN_MLU
    work->cncl_work_ =
        cncl_process_group_->allgather(outputTensors, inputTensors, opts);
    work->cncl_work_->wait();
    work->future_->markCompleted(work->cncl_work_->result());
#endif
#ifdef KAITIAN_CUDA
    work->nccl_work_ =
        nccl_process_group_->allgather(outputTensors, inputTensors, opts);
    work->nccl_work_->wait();
    work->future_->markCompleted(work->nccl_work_->result());
#endif
    return work;
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::allreduce(
    std::vector<at::Tensor>& tensors, const AllreduceOptions& opts) {
    auto work = c10::make_intrusive<WorkKaiTian>(tensors[0].device());
    work->operation_ = OperationType::ALLREDUCE;
#ifdef KAITIAN_MLU
    work->cncl_work_ = cncl_process_group_->allreduce(tensors, opts);
    work->cncl_work_->wait();
    work->future_->markCompleted(at::IValue(work->cncl_work_->result()));
#endif
#ifdef KAITIAN_CUDA
    work->nccl_work_ = nccl_process_group_->allreduce(tensors, opts);
    work->nccl_work_->wait();
    work->future_->markCompleted(work->nccl_work_->result());
#endif
    return work;
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
    auto work = c10::make_intrusive<WorkKaiTian>(tensors[0].device());
    work->operation_ = OperationType::BROADCAST;
#ifdef KAITIAN_MLU
    work->cncl_work_ = cncl_process_group_->broadcast(tensors, opts);
    work->cncl_work_->wait();
    work->future_->markCompleted(at::IValue(work->cncl_work_->result()));
#endif
#ifdef KAITIAN_CUDA
    work->nccl_work_ = nccl_process_group_->broadcast(tensors, opts);
    work->nccl_work_->wait();
    work->future_->markCompleted(work->nccl_work_->result());
#endif
    return work;
}

c10::intrusive_ptr<ProcessGroup> ProcessGroupKaiTian::createProcessGroupKaiTian(
    const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size,
    const std::chrono::duration<float>& /* unused */) {
    return c10::make_intrusive<ProcessGroupKaiTian>(store, rank, size);
}

}  // namespace c10d

extern void gloo_init(const std::string&);

PYBIND11_MODULE(_C, m) {
    m.def("createProcessGroupKaiTian",
          &c10d::ProcessGroupKaiTian::createProcessGroupKaiTian);
    m.def("gloo_init", &gloo_init);
}