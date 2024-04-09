#pragma once

#include "../../detail/prologue.hpp"

#include "detail/temporarily_with_current_device.hpp"
#include "detail/throw_on_error.hpp"
#include "event.hpp"
#include <cuda_runtime_api.h>
#include <utility>


namespace ubu::cuda
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

        detail::throw_on_error(cudaMalloc(&result, num_bytes), "cuda::device_memory_resource::allocate: after cudaMalloc");

        return result;
      });
    }

    inline void deallocate(void* ptr, std::size_t) const
    {
      detail::temporarily_with_current_device(device(), [=]
      {
        detail::throw_on_error(cudaFree(ptr), "cuda::device_memory_resource::deallocate: after cudaFree");
      });
    }

    inline std::pair<event,void*> allocate_after(const event& before, std::size_t num_bytes) const
    {
      return detail::temporarily_with_current_device(device(), [&]
      {
        detail::throw_on_error(cudaStreamWaitEvent(stream_, before.native_handle()),
          "cuda::device_memory_resource::allocate_after: after cudaStreamWaitEvent"
        );

        void* ptr{};
        detail::throw_on_error(cudaMallocAsync(reinterpret_cast<void**>(&ptr), num_bytes, stream()),
          "cuda::device_memory_resource::allocate_after: after cudaMallocAsync"
        );

        event after{device(), stream()};

        return std::pair<event,void*>(std::move(after), ptr);
      });
    }

    inline std::pair<event,void*> allocate_and_zero_after(const event& before, std::size_t num_bytes) const
    {
      auto [_, ptr] = allocate_after(before, num_bytes);

      detail::throw_on_error(cudaMemsetAsync(ptr, 0, num_bytes, stream_),
        "cuda::device_memory_resource::allocate_and_zero_after: after cudaMemsetAsync"
      );

      event after{device(), stream()};

      return std::pair<event,void*>(std::move(after), ptr);
    }

    inline event deallocate_after(const event& before, void* ptr, std::size_t) const
    {
      detail::throw_on_error(cudaStreamWaitEvent(stream(), before.native_handle()),
        "cuda::device_memory_resource::deallocate_after: after cudaStreamWaitEvent"
      );

      detail::throw_on_error(cudaFreeAsync(ptr, stream()),
        "cuda::device_memory_resource::deallocate_after: after cudaFreeAsync"
      );

      return {device_, stream()};
    }

    inline int device() const
    {
      return device_;
    }

    inline cudaStream_t stream() const
    {
      return stream_;
    }

    // returns the maximum size, in bytes, of the largest
    // theoretical allocation allocate could accomodate
    inline std::size_t max_size() const
    {
      return detail::temporarily_with_current_device(device(), [=]
      {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        detail::throw_on_error(cudaMemGetInfo(&free_bytes, &total_bytes), "cuda::managed_memory_resource::allocate: after cudaMemGetInfo");
        return free_bytes;
      });
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


} // end ubu::cuda


#include "../../detail/epilogue.hpp"

