#include <pybind11/chrono.h>

extern void init_scheduler(pybind11::module &);
extern void init_backend(pybind11::module &);
extern void init_device(pybind11::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    init_scheduler(m);
    init_backend(m);
    init_device(m);
}