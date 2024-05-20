#pragma once

#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/python.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

constexpr const char* BACKEND_NAME = "kaitian";

namespace c10d {

class ProcessGroupKaiTian : public ProcessGroup {
   public:
    ProcessGroupKaiTian(const c10::intrusive_ptr<c10d::Store>& store, int rank,
                        int size);

    c10::intrusive_ptr<Work> broadcast(
        std::vector<at::Tensor>& data,
        const BroadcastOptions& opts = BroadcastOptions()) override;

    c10::intrusive_ptr<Work> allreduce(
        std::vector<at::Tensor>& tensors,
        const AllreduceOptions& opts = AllreduceOptions()) override;

    c10::intrusive_ptr<Work> allreduce_coalesced(
        std::vector<at::Tensor>& tensors,
        const AllreduceCoalescedOptions& opts =
            AllreduceCoalescedOptions()) override;

    c10::intrusive_ptr<Work> reduce(
        std::vector<at::Tensor>& tensors,
        const ReduceOptions& opts = ReduceOptions()) override;

    c10::intrusive_ptr<Work> allgather(
        std::vector<std::vector<at::Tensor>>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const AllgatherOptions& opts = AllgatherOptions()) override;

    c10::intrusive_ptr<Work> _allgather_base(
        at::Tensor& outputBuffer, at::Tensor& inputBuffer,
        const AllgatherOptions& opts = AllgatherOptions()) override;

    c10::intrusive_ptr<Work> barrier(
        const BarrierOptions& opts = BarrierOptions()) override;

    c10::intrusive_ptr<Work> gather(
        std::vector<std::vector<at::Tensor>>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const GatherOptions& opts = GatherOptions()) override;

    c10::intrusive_ptr<Work> scatter(
        std::vector<at::Tensor>& outputTensors,
        std::vector<std::vector<at::Tensor>>& inputTensors,
        const ScatterOptions& opts = ScatterOptions()) override;

    c10::intrusive_ptr<Work> reduce_scatter(
        std::vector<at::Tensor>& outputTensors,
        std::vector<std::vector<at::Tensor>>& inputTensors,
        const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

    c10::intrusive_ptr<Work> alltoall_base(
        at::Tensor& outputTensor, at::Tensor& inputTensor,
        std::vector<int64_t>& outputSplitSizes,
        std::vector<int64_t>& inputSplitSizes,
        const AllToAllOptions& opts = AllToAllOptions()) override;

    c10::intrusive_ptr<Work> alltoall(
        std::vector<at::Tensor>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const AllToAllOptions& opts = AllToAllOptions()) override;

    c10::intrusive_ptr<Work> send(std::vector<at::Tensor>& tensors, int dstRank,
                                  int tag) override;

    c10::intrusive_ptr<Work> recv(std::vector<at::Tensor>& tensors, int srcRank,
                                  int tag) override;

    c10::intrusive_ptr<Work> recvAnysource(std::vector<at::Tensor>& tensors,
                                           int tag) override;

    static c10::intrusive_ptr<ProcessGroup> createProcessGroupKaiTian(
        const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size,
        const std::chrono::duration<float>& timeout);

    static void ProcessGroupKaiTianConstructor() __attribute__((constructor)) {
        py::object module = py::module::import("torch.distributed");
        py::object register_backend =
            module.attr("Backend").attr("register_backend");
        register_backend("kaitian",
                         py::cpp_function(createProcessGroupKaiTian));
    }

    const std::string getBackendName() const override {
        return std::string(BACKEND_NAME);
    }

   private:
    c10::intrusive_ptr<Store> store_;
#ifdef SUPPORT_CAMBRICON_MLU
    c10::intrusive_ptr<ProcessGroup> cncl_process_group_;
#endif
};

class WorkKaiTian : public Work {
    friend class ProcessGroupKaiTian;

   public:
    WorkKaiTian(at::Device device);
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
    virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

   private:
    c10::intrusive_ptr<at::ivalue::Future> future_;
    at::Device device_;
#ifdef SUPPORT_CAMBRICON_MLU
    c10::intrusive_ptr<Work> cncl_work_;
#endif
};

}  // namespace c10d
