This directory contains functionality related to reflecting on the current program environment

* `has_cuda_runtime.hpp` - Defines the `detail::has_cuda_runtime` function, which is used to determine whether CUDA Runtime API calls are available to the current target.
* `is_device.hpp` - Defines the `detail::is_device` function, which is used to determine whether the current target is a CUDA device GPU.
* `is_host.hpp` - Defines the `detail::is_host` function, which is used to determine whether the current target is a CUDA host CPU.

