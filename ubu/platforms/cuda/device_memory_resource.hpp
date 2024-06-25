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
    device_memory_resource(int device, cudaStream_t stream, cudaMemPool_t pool)
      : device_{device},
        stream_{stream},
        pool_{pool}
    {}

    device_memory_resource(int device, cudaStream_t stream)
      : device_memory_resource(device, stream, default_pool(device))
    {}

    device_memory_resource()
      : device_memory_resource{0,0}
    {}

    device_memory_resource(const device_memory_resource&) = default;

    inline void* allocate(std::size_t num_bytes) const
    {
      // use the asynchronous method
      cuda::event before(device(), stream());
      auto [after,result] = allocate_after(before, num_bytes);
      after.wait();
      return result;
    }

    inline void deallocate(void* ptr, std::size_t num_bytes) const
    {
      // use the asynchronous method
      cuda::event before(device(), stream());
      cuda::event after = deallocate_after(before, ptr, num_bytes);
      after.wait();
    }

    inline std::pair<event,void*> allocate_after(const event& before, std::size_t num_bytes) const
    {
      return detail::temporarily_with_current_device(device(), [&]
      {
        detail::throw_on_error(cudaStreamWaitEvent(stream_, before.native_handle()),
          "cuda::device_memory_resource::allocate_after: after cudaStreamWaitEvent"
        );

        void* ptr{};
        detail::throw_on_error(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&ptr), num_bytes, pool(), stream()),
          "cuda::device_memory_resource::allocate_after: after cudaMallocFromPoolAsync"
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

    inline cudaMemPool_t pool() const
    {
      return pool_;
    }

    // returns the maximum size, in bytes, of the largest
    // theoretical allocation allocate could accomodate
    inline std::size_t max_size() const
    {
      // XXX we should probably report a value based on our memory pool,
      //     rather than the total size of memory
      return detail::temporarily_with_current_device(device(), [=]
      {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        detail::throw_on_error(cudaMemGetInfo(&free_bytes, &total_bytes), "cuda::managed_memory_resource::allocate: after cudaMemGetInfo");
        return free_bytes;
      });
    }

    // enables the given peer device to read and write memory
    // in this device_memory_resource's pool
    inline void enable_access(int peer_device) const
    {
      cudaMemAccessDesc access;
      access.location.type = cudaMemLocationTypeDevice;
      access.location.id = peer_device;
      access.flags = cudaMemAccessFlagsProtReadWrite;

      detail::throw_on_error(cudaMemPoolSetAccess(pool(), &access, 1),
        "cuda::device_memory_resource::enable_access: after cudaMemPoolSetAccess"
      );
    }

    inline bool is_equal(const device_memory_resource& other) const
    {
      return device() == other.device() and stream() == other.stream() and pool() == other.pool();
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
    // returns the default pool on the given device
    inline static cudaMemPool_t default_pool(int device)
    {
      cudaMemPool_t result{};
      detail::throw_on_error(cudaDeviceGetDefaultMemPool(&result, device),
        "cuda::device_memory_resource::default_pool: after cudaGetDefaultMemPool"
      );

      return result;
    }

    int device_;
    cudaStream_t stream_;
    cudaMemPool_t pool_;
};


} // end ubu::cuda


#include "../../detail/epilogue.hpp"

