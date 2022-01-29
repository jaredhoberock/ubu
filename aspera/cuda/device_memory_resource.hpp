#pragma once

#include "../detail/prologue.hpp"

#include "../detail/exception.hpp"
#include "detail/temporarily_with_current_device.hpp"
#include <cuda_runtime_api.h>

ASPERA_NAMESPACE_OPEN_BRACE


namespace cuda
{


class device_memory_resource
{
  public:
    inline explicit device_memory_resource(int device)
      : device_{device}
    {}

    inline device_memory_resource()
      : device_memory_resource{0}
    {}

    device_memory_resource(const device_memory_resource&) = default;

    inline void* allocate(std::size_t num_bytes) const
    {
      return detail::temporarily_with_current_device(device(), [=]
      {
        void* result = nullptr;

        detail::throw_on_error(cudaMalloc(&result, num_bytes), "cuda::device_memory_resource::allocate: CUDA error after cudaMalloc");

        return result;
      });
    }

    inline void deallocate(void* ptr, std::size_t) const
    {
      detail::temporarily_with_current_device(device(), [=]
      {
        detail::throw_on_error(cudaFree(ptr), "cuda::device_memory_resource::deallocate: CUDA error after cudaFree");
      });
    }

    inline int device() const
    {
      return device_;
    }

    inline bool is_equal(const device_memory_resource& other) const
    {
      return device() == other.device();
    }

    inline bool operator==(const device_memory_resource& other) const
    {
      return is_equal(other);
    }

    inline bool operator!=(const device_memory_resource& other) const
    {
      return !(*this == other);
    }

  private:
    int device_;
};


} // end cuda


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

