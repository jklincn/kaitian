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

    c10::intrusive_ptr<Work> allgather(
        std::vector<std::vector<at::Tensor>>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const AllgatherOptions& opts = AllgatherOptions()) override;

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
#ifdef KAITIAN_MLU
    c10::intrusive_ptr<ProcessGroup> cncl_process_group_;
#endif
#ifdef KAITIAN_CUDA
    c10::intrusive_ptr<ProcessGroup> nccl_process_group_;
#endif
};

typedef enum { ALLGATHER, ALLREDUCE, BROADCAST } OperationType;

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
    OperationType operation_;
#ifdef KAITIAN_MLU
    c10::intrusive_ptr<Work> cncl_work_;
#endif
#ifdef KAITIAN_CUDA
    c10::intrusive_ptr<Work> nccl_work_;
#endif
};

}  // namespace c10d
