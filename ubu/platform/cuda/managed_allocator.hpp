#pragma once

#include "../../detail/prologue.hpp"

#include "managed_memory_resource.hpp"
#include <cuda_runtime_api.h>

namespace ubu::cuda
{


template<class T>
class managed_allocator : private managed_memory_resource
{
  private:
    using super_t = managed_memory_resource;

  public:
    using value_type = T;

    explicit managed_allocator(int device)
      : super_t{device}
    {}

    managed_allocator()
      : managed_allocator{0}
    {}

    managed_allocator(const managed_allocator&) = default;

    template<class OtherU>
    managed_allocator(const managed_allocator<OtherU>& other)
      : managed_allocator{other.device()}
    {}

    T* allocate(std::size_t n) const
    {
      return reinterpret_cast<T*>(super_t::allocate(sizeof(T) * n));
    }

    void deallocate(T* ptr, std::size_t n) const
    {
      super_t::deallocate(ptr, sizeof(T) * n);
    }

    int device() const
    {
      return super_t::device();
    }

    // returns the maximum size, in elements, of the largest
    // theoretical allocation allocate could accomodate
    inline std::size_t max_size() const
    {
      return super_t::max_size() / sizeof(T);
    }

    bool operator==(const managed_allocator& other) const
    {
      return super_t::operator==(other);
    }

    inline bool operator!=(const managed_allocator& other) const
    {
      return !(*this == other);
    }
};


} // end ubu::cuda


#include "../../detail/epilogue.hpp"

