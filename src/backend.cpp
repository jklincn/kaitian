#include "backend.hpp"

#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>

#include <vector>

#include "support/cambricon_mlu.hpp"

namespace c10d {
WorkKaiTian::WorkKaiTian(at::Device device) : device_(device) {
    future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()),
        std::vector<at::Device>{device});
}
bool WorkKaiTian::isCompleted() { return true; }

bool WorkKaiTian::isSuccess() const { return true; }

bool WorkKaiTian::wait(std::chrono::milliseconds) { return true; }

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

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_tensor_list(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other, size_t world_size) {
    if (tensor_lists.size() != 1 || other.size() != 1) {
        throw std::runtime_error(
            "MLU Tensors must be on a single MLU device per process");
    }

    if (tensor_lists[0].size() == 0) {
        throw std::runtime_error("Received an empty list");
    }

    if (tensor_lists[0].size() != world_size) {
        throw std::runtime_error(
            "Tensor list input to scatter/gather must match number of "
            "collective"
            " participants");
    }

    auto device = other[0].device();
    for (const auto& t : tensor_lists[0]) {
        if (t.numel() != other[0].numel()) {
            throw std::runtime_error(
                "All tensor operands to scatter/gather must have the same "
                "number of elements");
        }
        if (t.device() != device) {
            throw std::runtime_error(
                "Expecting all tensors on the same device");
        }
    }

    auto& t = tensor_lists[0][0];
    std::vector<int64_t> new_size{static_cast<int64_t>(tensor_lists[0].size())};
    std::vector<int64_t> new_stride{t.numel()};
    new_size.insert(new_size.end(), t.sizes().begin(), t.sizes().end());
    new_stride.insert(new_stride.end(), t.strides().begin(), t.strides().end());
    return {at::empty_strided(new_size, new_stride,
                              t.options().memory_format(c10::nullopt))};
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const AllgatherOptions& opts) {
    at::Device device = outputTensors[0][0].device();
    auto work = c10::make_intrusive<WorkKaiTian>(device);
#ifdef SUPPORT_CAMBRICON_MLU
    work->cncl_work_ =
        cncl_process_group_->allgather(outputTensors, inputTensors, opts);
    work->cncl_work_->wait();
#endif
    std::vector<at::Tensor> output_flattened =
        flatten_tensor_list(outputTensors, inputTensors, size_);
    work->future_->markCompleted(at::IValue(output_flattened));
    return work;
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::_allgather_base(
    at::Tensor& /* unused */, at::Tensor& /* unused */,
    const AllgatherOptions& /* unused */) {
    throw std::runtime_error("[kaitian] error: not supported _allgather_base");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::allreduce(
    std::vector<at::Tensor>& tensors, const AllreduceOptions& opts) {
    auto work = c10::make_intrusive<WorkKaiTian>(tensors[0].device());
#ifdef SUPPORT_CAMBRICON_MLU
    work->cncl_work_ = cncl_process_group_->allreduce(tensors, opts);
    work->cncl_work_->wait();
#endif
    work->future_->markCompleted(at::IValue(tensors));
    return work;
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */) {
    throw std::runtime_error(
        "[kaitian] error: not supported allreduce_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::alltoall(
    std::vector<at::Tensor>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllToAllOptions& /* unused */) {
    throw std::runtime_error("[kaitian] error: not supported alltoall");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::alltoall_base(
    at::Tensor& outputTensor, at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
    throw std::runtime_error("[kaitian] error: not supported alltoall_base");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::barrier(
    const BarrierOptions& /* unused */) {
    throw std::runtime_error("[kaitian] error: not supported barrier");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
    auto work = c10::make_intrusive<WorkKaiTian>(tensors[0].device());
#ifdef SUPPORT_CAMBRICON_MLU
    work->cncl_work_ = cncl_process_group_->broadcast(tensors, opts);
    work->cncl_work_->wait();
#endif
    work->future_->markCompleted(at::IValue(tensors));
    return work;
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */, const GatherOptions& /* unused */) {
    throw std::runtime_error("[kaitian] error: not supported gather");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::reduce(
    std::vector<at::Tensor>& /* unused */, const ReduceOptions& /* unused */) {
    throw std::runtime_error("[kaitian] error: not supported reduce");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */) {
    throw std::runtime_error("[kaitian] error: not supported reduce_scatter");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
    throw std::runtime_error("[kaitian] error: not supported scatter");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::send(
    std::vector<at::Tensor>& tensors, int dstRank, int tag) {
    throw std::runtime_error("[kaitian] error: not supported send");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::recv(
    std::vector<at::Tensor>& tensors, int srcRank, int tag) {
    throw std::runtime_error("[kaitian] error: not supported recv");
}

c10::intrusive_ptr<Work> ProcessGroupKaiTian::recvAnysource(
    std::vector<at::Tensor>& tensors, int tag) {
    throw std::runtime_error("[kaitian] error: not supported recvAnysource");
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