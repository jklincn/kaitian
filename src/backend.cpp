#include "backend.hpp"

#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>

#include <iostream>
#include <vector>

#include "support/support.hpp"

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

bool WorkKaiTian::wait(std::chrono::milliseconds) {
    return true;
    //     if (operation_ == OperationType::ALLREDUCE) {
    //         return true;
    //     }
    //     // std::cout << "wait, device: " << device_.str()
    //     //           << ", operation: " << operation_ << std::endl;
    // #ifdef SUPPORT_CAMBRICON_MLU
    //     if (cncl_work_ != nullptr) {
    //         // std::cout << "wait" << std::endl;
    //         cncl_work_->wait();
    //         // std::cout << "wait end" << std::endl;
    //     }
    // #endif
    //     // std::cout << "markCompleted" << std::endl;
    //     future_->markCompleted(at::IValue(cncl_work_->result()));
    //     return true;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkKaiTian::getFuture() {
    return future_;
}

ProcessGroupKaiTian::ProcessGroupKaiTian(
    const c10::intrusive_ptr<c10d::Store>& store, int rank, int size)
    : ProcessGroup(rank, size), store_(store) {
#ifdef SUPPORT_CAMBRICON_MLU
    cncl_process_group_ = torch_mlu::ProcessGroupCNCL::createProcessGroupCNCL(
        store, rank, size, std::chrono::seconds(60));
#endif
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const AllgatherOptions& opts) {
    // std::cout << "allgather, device: " << outputTensors[0][0].device().str()
    //           << std::endl;
    auto work = c10::make_intrusive<WorkKaiTian>(outputTensors[0][0].device());
    work->operation_ = OperationType::ALLGATHER;
#ifdef SUPPORT_CAMBRICON_MLU
    work->cncl_work_ =
        cncl_process_group_->allgather(outputTensors, inputTensors, opts);
    work->cncl_work_->wait();
    work->future_->markCompleted(work->cncl_work_->result());
#endif

    return work;
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::allreduce(
    std::vector<at::Tensor>& tensors, const AllreduceOptions& opts) {
    // std::cout << "allreduce, device: " << tensors[0].device().str()
    //           << std::endl;
    auto work = c10::make_intrusive<WorkKaiTian>(tensors[0].device());
    work->operation_ = OperationType::ALLREDUCE;
#ifdef SUPPORT_CAMBRICON_MLU
    work->cncl_work_ = cncl_process_group_->allreduce(tensors, opts);
    work->cncl_work_->wait();
    work->future_->markCompleted(at::IValue(work->cncl_work_->result()));
#endif

    // std::cout << "allreduce, device: " << tensors[0].device().str() << "out"
    //           << std::endl;
    return work;
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
    // std::cout << "broadcast, device: " << tensors[0].device().str()
    //           << std::endl;
    auto work = c10::make_intrusive<WorkKaiTian>(tensors[0].device());
    work->operation_ = OperationType::BROADCAST;
#ifdef SUPPORT_CAMBRICON_MLU
    work->cncl_work_ = cncl_process_group_->broadcast(tensors, opts);
    work->cncl_work_->wait();
    work->future_->markCompleted(at::IValue(work->cncl_work_->result()));
#endif

    return work;
}

c10::intrusive_ptr<ProcessGroup> ProcessGroupKaiTian::createProcessGroupKaiTian(
    const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size,
    const std::chrono::duration<float>& /* unused */) {
    return c10::make_intrusive<ProcessGroupKaiTian>(store, rank, size);
}

}  // namespace c10d

void init_backend(pybind11::module& m) {
    m.def("createProcessGroupKaiTian",
          &c10d::ProcessGroupKaiTian::createProcessGroupKaiTian);
}