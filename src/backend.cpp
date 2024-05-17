#include "backend.hpp"

#include "support/cambricon_mlu.hpp"

namespace c10d {

bool WorkKaiTian::isCompleted() { return true; }

bool WorkKaiTian::isSuccess() const { return true; }

bool WorkKaiTian::wait(std::chrono::milliseconds /* unused */) { return true; }

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
#ifdef SUPPORT_CAMBRICON_MLU
    return cncl_process_group_->allgather(outputTensors, inputTensors, opts);
#endif
    throw std::runtime_error("allgather: no available devices");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::_allgather_base(
    at::Tensor& /* unused */, at::Tensor& /* unused */,
    const AllgatherOptions& /* unused */) {
    throw std::runtime_error("not supported2");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::allreduce(
    std::vector<at::Tensor>& tensors, const AllreduceOptions& opts) {
#ifdef SUPPORT_CAMBRICON_MLU
    return cncl_process_group_->allreduce(tensors, opts);
#endif
    throw std::runtime_error("allreduce: no available devices");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */) {
    throw std::runtime_error("not supported3");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::alltoall(
    std::vector<at::Tensor>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllToAllOptions& /* unused */) {
    throw std::runtime_error("not supported4");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::alltoall_base(
    at::Tensor& outputTensor, at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
    throw std::runtime_error("not supported5");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::barrier(
    const BarrierOptions& /* unused */) {
    throw std::runtime_error("not supported6");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
#ifdef SUPPORT_CAMBRICON_MLU
    return cncl_process_group_->broadcast(tensors, opts);
#endif
    throw std::runtime_error("broadcast: no available devices");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */, const GatherOptions& /* unused */) {
    throw std::runtime_error("not supported7");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::reduce(
    std::vector<at::Tensor>& /* unused */, const ReduceOptions& /* unused */) {
    throw std::runtime_error("not supported8");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */) {
    throw std::runtime_error("not supported9");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
    throw std::runtime_error("not supported10");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::send(
    std::vector<at::Tensor>& tensors, int dstRank, int tag) {
    throw std::runtime_error("not supported11");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::recv(
    std::vector<at::Tensor>& tensors, int srcRank, int tag) {
    throw std::runtime_error("not supported12");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::recvAnysource(
    std::vector<at::Tensor>& tensors, int tag) {
    throw std::runtime_error("not supported13");
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