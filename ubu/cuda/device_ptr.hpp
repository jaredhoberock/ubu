#pragma once

#include "../detail/prologue.hpp"

#include "../detail/exception.hpp"
#include "../memory/plain_old_data.hpp"
#include "../memory/pointer/fancy_ptr.hpp"
#include "detail/temporarily_with_current_device.hpp"
#include "detail/throw_on_error.hpp"
#include <cassert>
#include <concepts>
#include <cstring>
#include <type_traits>


namespace ubu::cuda
{


class device_memory_copier
{
  public:
    template<plain_old_data_or_void T>
    using address = T*;

    constexpr device_memory_copier(int device)
      : device_{device}
    {}

    constexpr device_memory_copier()
      : device_memory_copier{0}
    {}

    device_memory_copier(const device_memory_copier&) = default;

    template<plain_old_data T>
    void copy_n(const T* from, std::size_t count, T* to) const
    {
#if defined(__CUDACC__)
      if UBU_TARGET(ubu::detail::is_host())
      {
        detail::temporarily_with_current_device(device_, [=]
        {
          detail::throw_on_error(cudaMemcpy(to, from, sizeof(T) * count, cudaMemcpyDefault),
            "device_memory_copier: after cudaMemcpy"
          );
        });
      }
      else if UBU_TARGET(ubu::detail::is_device())
      {
        std::memcpy(to, from, sizeof(T) * count);
      }
      else
      {
        // this should never be reached
        assert(0);
      }
#else
      detail::temporarily_with_current_device(device_, [=]
      {
        detail::throw_on_error(cudaMemcpy(to, from, sizeof(T) * count, cudaMemcpyDefault),
          "device_memory_copier: after cudaMemcpy"
        );
      });
#endif
    }

    constexpr int device() const
    {
      return device_;
    }

    constexpr bool operator==(const device_memory_copier& other) const
    {
      return device() == other.device();
    }

  private:
    int device_;
};


template<plain_old_data_or_void T>
using device_ptr = fancy_ptr<T, device_memory_copier>;


} // end ubu::cuda

#include "../detail/epilogue.hpp"

