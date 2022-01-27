#pragma once

#include "../detail/prologue.hpp"

#include "../detail/exception.hpp"
#include "detail/temporarily_with_current_device.hpp"
#include <cuda_runtime_api.h>

ASPERA_NAMESPACE_OPEN_BRACE


namespace cuda
{


class managed_memory_resource
{
  public:
    inline explicit managed_memory_resource(int device)
      : device_{device}
    {}

    inline managed_memory_resource()
      : managed_memory_resource{0}
    {}

    managed_memory_resource(const managed_memory_resource&) = default;

    inline void* allocate(std::size_t num_bytes) const
    {
      return detail::temporarily_with_current_device(device(), [=]
      {
        void* result = nullptr;

        detail::throw_on_error(cudaMallocManaged(&result, num_bytes, cudaMemAttachGlobal), "cuda::managed_memory_resource::allocate: CUDA error after cudaMallocManaged");

        return result;
      });
    }

    inline void deallocate(void* ptr, std::size_t) const
    {
      detail::temporarily_with_current_device(device(), [=]
      {
        detail::throw_on_error(cudaFree(ptr), "cuda::managed_resource::deallocate: CUDA error after cudaFree");
      });
    }

    inline int device() const
    {
      return device_;
    }

    inline bool is_equal(const managed_memory_resource& other) const
    {
      return device() == other.device();
    }

    inline bool operator==(const managed_memory_resource& other) const
    {
      return is_equal(other);
    }

    inline bool operator!=(const managed_memory_resource& other) const
    {
      return !(*this == other);
    }

  private:
    int device_;
};


} // end cuda


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

