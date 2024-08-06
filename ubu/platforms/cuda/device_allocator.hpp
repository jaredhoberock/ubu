#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/exception.hpp"
#include "../../detail/reflection.hpp"
#include "device_memory_resource.hpp"
#include "device_ptr.hpp"
#include "event.hpp"
#include <cstddef>
#include <cuda_runtime_api.h>
#include <utility>


namespace ubu::cuda
{


// XXX eliminate device_memory_resource


template<class T>
class device_allocator : private device_memory_resource
{
  private:
    using super_t = device_memory_resource;

  public:
    using value_type = T;
    using pointer = device_ptr<T>;
    using happening_type = event;

    device_allocator(int device, cudaStream_t stream, cudaMemPool_t pool)
      : super_t{device, stream, pool}
    {}

    device_allocator(int device, cudaStream_t stream)
      : super_t{device, stream}
    {}

    explicit device_allocator(int device)
      : device_allocator{device, 0}
    {}

    device_allocator()
      : device_allocator{0}
    {}

    device_allocator(const device_allocator&) = default;

    template<class U>
    device_allocator(const device_allocator<U>& other)
      : device_allocator{other.device(), other.stream()}
    {}

    pointer allocate(std::size_t n) const
    {
      T* raw_ptr = reinterpret_cast<T*>(super_t::allocate(sizeof(T) * n));
      return {raw_ptr, device(), stream()};
    }

    void deallocate(pointer ptr, std::size_t n) const
    {
      super_t::deallocate(ptr.to_address(), sizeof(T) * n);
    }

    std::pair<event, device_ptr<T>> allocate_after(const event& before, std::size_t n) const
    {
      auto [allocation_ready, raw_ptr] = super_t::allocate_after(before, sizeof(T) * n);
      device_ptr<T> d_ptr{reinterpret_cast<T*>(raw_ptr), device(), stream()};

      return {std::move(allocation_ready), d_ptr};
    }

    std::pair<event, device_ptr<T>> allocate_and_zero_after(const event& before, std::size_t n) const
    {
      auto [allocation_ready, raw_ptr] = super_t::allocate_and_zero_after(before, sizeof(T) * n);
      device_ptr<T> d_ptr{reinterpret_cast<T*>(raw_ptr), device(), stream()};

      return {std::move(allocation_ready), d_ptr};
    }

    event deallocate_after(const event& before, pointer ptr, std::size_t n) const
    {
      return super_t::deallocate_after(before, ptr.to_address(), sizeof(T) * n);
    }

    int device() const
    {
      return super_t::device();
    }

    cudaStream_t stream() const
    {
      return super_t::stream();
    }

    cudaMemPool_t pool() const
    {
      return super_t::pool();
    }

    // returns the maximum size, in elements, of the largest
    // theoretical allocation allocate could accomodate
    inline std::size_t max_size() const
    {
      return super_t::max_size() / sizeof(T);
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


device_allocator(int) -> device_allocator<std::byte>;


} // end ubu::cuda

#include "../../detail/epilogue.hpp"

