#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/exception.hpp"
#include "../../detail/reflection.hpp"
#include "device_memory_resource.hpp"
#include "device_ptr.hpp"
#include "event.hpp"
#include <cuda_runtime_api.h>
#include <utility>


namespace ubu::cuda
{


// XXX eliminate device_memory_resource


template<class T>
class temporary_allocator : private device_memory_resource
{
  private:
    using super_t = device_memory_resource;

  public:
    using value_type = T;
    using pointer = device_ptr<T>;
    using happening_type = event;

    constexpr temporary_allocator(int device, cudaStream_t s)
      : super_t{device, s}
    {}

    explicit constexpr temporary_allocator(int device)
      : temporary_allocator{device, 0}
    {}

    constexpr temporary_allocator()
      : temporary_allocator{0}
    {}

    temporary_allocator(const temporary_allocator&) = default;

    template<class OtherU>
    constexpr temporary_allocator(const temporary_allocator<OtherU>& other)
      : temporary_allocator{other.device()}
    {}

    constexpr pointer allocate(std::size_t n) const
    {
      // XXX this should suballocate from a cache of small allocations

      T* raw_ptr = reinterpret_cast<T*>(super_t::allocate(sizeof(T) * n));
      return {raw_ptr, device()};
    }

    constexpr void deallocate(pointer ptr, std::size_t n) const
    {
      super_t::deallocate(ptr.to_address(), sizeof(T) * n);
    }

    constexpr std::pair<event, device_ptr<T>> allocate_after(const event& before, std::size_t n) const
    {
      // XXX this should suballocate from a cache of small allocations

      auto [allocation_ready, raw_ptr] = super_t::allocate_after(before, sizeof(T) * n);
      device_ptr<T> d_ptr{reinterpret_cast<T*>(raw_ptr), device()};

      return {std::move(allocation_ready), d_ptr};
    }

    event deallocate_after(const event& before, pointer ptr, std::size_t n) const
    {
      return super_t::deallocate_after(before, ptr.to_address(), sizeof(T) * n);
    }

    constexpr int device() const
    {
      return super_t::device();
    }

    constexpr cudaStream_t stream() const
    {
      return super_t::stream();
    }

    constexpr bool operator==(const temporary_allocator& other) const
    {
      return super_t::operator==(other);
    }

    constexpr bool operator!=(const temporary_allocator& other) const
    {
      return !(*this == other);
    }
};


} // end ubu::cuda

#include "../../detail/epilogue.hpp"

