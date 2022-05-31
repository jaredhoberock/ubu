#pragma once

#include "../detail/prologue.hpp"

#include "../detail/exception.hpp"
#include "../detail/reflection.hpp"
#include "device_memory_resource.hpp"
#include "device_ptr.hpp"
#include "event.hpp"
#include "kernel_executor.hpp"
#include <cuda_runtime_api.h>
#include <utility>

UBU_NAMESPACE_OPEN_BRACE


namespace cuda
{


template<class T>
class device_allocator : private device_memory_resource
{
  private:
    using super_t = device_memory_resource;

  public:
    using value_type = T;
    using pointer = device_ptr<T>;

    // this typedef is a requirement of asynchronous_deleter
    using event_type = cuda::event;

    device_allocator(int device, cudaStream_t s)
      : super_t{device, s}
    {}

    explicit device_allocator(int device)
      : device_allocator{device, 0}
    {}

    device_allocator()
      : device_allocator{0}
    {}

    device_allocator(const device_allocator&) = default;

    template<class OtherU>
    device_allocator(const device_allocator<OtherU>& other)
      : device_allocator{other.device()}
    {}

    const kernel_executor associated_executor() const
    {
      return {device(), stream()};
    }

    pointer allocate(std::size_t n) const
    {
      T* raw_ptr = reinterpret_cast<T*>(super_t::allocate(sizeof(T) * n));
      return {raw_ptr, device()};
    }

    void deallocate(pointer ptr, std::size_t n) const
    {
      super_t::deallocate(ptr.native_handle(), sizeof(T) * n);
    }

    std::pair<event_type, device_ptr<T>> allocate_after(const event& before, std::size_t n) const
    {
      auto [allocation_ready, raw_ptr] = super_t::allocate_after(before, sizeof(T) * n);
      device_ptr<T> d_ptr{reinterpret_cast<T*>(raw_ptr), device()};

      return {std::move(allocation_ready), d_ptr};
    }

    event deallocate_after(const event& before, pointer ptr, std::size_t n) const
    {
      return super_t::deallocate_after(before, ptr.native_handle(), sizeof(T) * n);
    }

    int device() const
    {
      return super_t::device();
    }

    cudaStream_t stream() const
    {
      return super_t::stream();
    }

    bool operator==(const device_allocator& other) const
    {
      return super_t::operator==(other);
    }

    bool operator!=(const device_allocator& other) const
    {
      return !(*this == other);
    }
};


} // end cuda


UBU_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

