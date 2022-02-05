#pragma once

#include "../detail/prologue.hpp"

#include "../detail/exception.hpp"
#include "detail/temporarily_with_current_device.hpp"
#include "event.hpp"
#include <cuda_runtime_api.h>
#include <utility>

ASPERA_NAMESPACE_OPEN_BRACE


namespace cuda
{


class device_memory_resource
{
  public:
    explicit device_memory_resource(int device, cudaStream_t stream)
      : device_{device},
        stream_{stream}
    {}

    inline device_memory_resource()
      : device_memory_resource{0,0}
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

    inline std::pair<event,void*> allocate_after(const event& before, std::size_t num_bytes) const
    {
      return detail::temporarily_with_current_device(device(), [&]
      {
        detail::throw_on_error(cudaStreamWaitEvent(stream_, before.native_handle()),
          "cuda::device_memory_resource::allocate_after: CUDA error after cudaStreamWaitEvent"
        );

        void* ptr{};
        detail::throw_on_error(cudaMallocAsync(reinterpret_cast<void**>(&ptr), num_bytes, stream()),
          "cuda::device_memory_resource::allocate_after: CUDA error after cudaMallocAsync"
        );

        event after{stream()};

        return std::pair<event,void*>(std::move(after), ptr);
      });
    }

    inline event deallocate_after(const event& before, void* ptr, std::size_t) const
    {
      detail::throw_on_error(cudaStreamWaitEvent(stream(), before.native_handle()),
        "cuda::device_memory_resource::deallocate_after: CUDA error after cudaStreamWaitEvent"
      );

      detail::throw_on_error(cudaFreeAsync(ptr, stream()),
        "cuda::device_memory_resource::deallocate_after: CUDA error after cudaFreeAsync"
      );

      return {stream()};
    }

    inline int device() const
    {
      return device_;
    }

    inline cudaStream_t stream() const
    {
      return stream_;
    }

    inline bool is_equal(const device_memory_resource& other) const
    {
      return device() == other.device() and stream() == other.stream();
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
    cudaStream_t stream_;
};


} // end cuda


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

