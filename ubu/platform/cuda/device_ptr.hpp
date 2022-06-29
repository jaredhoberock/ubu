#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/exception.hpp"
#include "../../memory/plain_old_data.hpp"
#include "../../memory/pointer/remote_ptr.hpp"
#include "detail/temporarily_with_current_device.hpp"
#include "detail/throw_on_error.hpp"
#include <cassert>
#include <concepts>
#include <cstring>
#include <type_traits>


namespace ubu::cuda
{


class device_memory_loader
{
  public:
    using address_type = void*;

    constexpr device_memory_loader(int device)
      : device_{device}
    {}

    constexpr device_memory_loader()
      : device_memory_loader{0}
    {}

    device_memory_loader(const device_memory_loader&) = default;

    void download(address_type from, std::size_t num_bytes, void* to) const
    {
#if defined(__CUDACC__)
      if UBU_TARGET(ubu::detail::is_host())
      {
        detail::temporarily_with_current_device(device_, [=]
        {
          detail::throw_on_error(cudaMemcpy(to, from, num_bytes, cudaMemcpyDeviceToHost),
            "device_memory_loader::download: after cudaMemcpy"
          );
        });
      }
      else if UBU_TARGET(ubu::detail::is_device())
      {
        std::memcpy(to, from, num_bytes);
      }
      else
      {
        // this should never be reached
        assert(0);
      }
#else
      detail::temporarily_with_current_device(device_, [=]
      {
        detail::throw_on_error(cudaMemcpy(to, from, num_bytes, cudaMemcpyDeviceToHost),
          "device_memory_loader::download: after cudaMemcpy"
        );
      });
#endif
    }

    void upload(const void* from, std::size_t num_bytes, address_type to) const
    {
#if defined(__CUDACC__)
      if UBU_TARGET(ubu::detail::is_host())
      {
        detail::temporarily_with_current_device(device_, [=]
        {
          detail::throw_on_error(cudaMemcpy(to, from, num_bytes, cudaMemcpyHostToDevice),
            "device_memory_loader::upload: after cudaMemcpy"
          );
        });
      }
      else if UBU_TARGET(ubu::detail::is_device())
      {
        std::memcpy(to, from, num_bytes);
      }
      else
      {
        // this should never be reached
        assert(0);
      }
#else
      detail::temporarily_with_current_device(device_, [=]
      {
        detail::throw_on_error(cudaMemcpy(to, from, num_bytes, cudaMemcpyHostToDevice),
          "device_memory_loader::upload after cudaMemcpy"
        );
      });
#endif
    }

    constexpr int device() const
    {
      return device_;
    }

    constexpr bool operator==(const device_memory_loader& other) const
    {
      return device() == other.device();
    }

  private:
    int device_;
};


template<plain_old_data_or_void T>
using device_ptr = remote_ptr<T, device_memory_loader>;


} // end ubu::cuda

#include "../../detail/epilogue.hpp"

