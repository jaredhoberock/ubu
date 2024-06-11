#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/exception.hpp"
#include "../../places/causality/initial_happening.hpp"
#include "../../places/causality/wait.hpp"
#include "../../places/memory/plain_old_data.hpp"
#include "../../places/memory/pointers/remote_ptr.hpp"
#include "detail/temporarily_with_current_device.hpp"
#include "detail/throw_on_error.hpp"
#include "event.hpp"
#include <cassert>
#include <concepts>
#include <cstring>
#include <type_traits>


namespace ubu::cuda
{


class device_memory_loader
{
  public:
    using happening_type = event;
    using address_type = void*;

    constexpr device_memory_loader(int device, cudaStream_t stream)
      : device_{device},
        stream_{stream}
    {}

    constexpr device_memory_loader(int device)
      : device_memory_loader{device, cudaStream_t{}}
    {}

    constexpr device_memory_loader()
      : device_memory_loader{0, cudaStream_t{}}
    {}

    device_memory_loader(const device_memory_loader&) = default;

    event download_after(const event& before, address_type from, std::size_t num_bytes, void* to) const
    {
#if defined(__CUDACC__)
      if UBU_TARGET(ubu::detail::is_host())
      {
        detail::throw_on_error(cudaStreamWaitEvent(stream(), before.native_handle()),
          "device_memory_loader::download_after: after streamWaitEvent"
        );

        detail::temporarily_with_current_device(device(), [&]
        {
          detail::throw_on_error(cudaMemcpyAsync(to, from, num_bytes, cudaMemcpyDeviceToHost, stream()),
            "device_memory_loader::download_after: after cudaMemcpy"
          );
        });

        return {device(), stream()};
      }
      else if UBU_TARGET(ubu::detail::is_device())
      {
        // XXX we need to wait on the before event somehow
        std::memcpy(to, from, num_bytes);
        return event(device(), stream());
      }
      else
      {
        // this should never be reached
        assert(0);
        return event(device(), stream());
      }
#else
      return detail::temporarily_with_current_device(device(), [&]
      {
        detail::throw_on_error(cudaStreamWaitEvent(stream(), before.native_handle()),
          "device_memory_loader::download_after: after cudaStreamWaitEvent"
        );

        detail::throw_on_error(cudaMemcpyAsync(to, from, num_bytes, cudaMemcpyDeviceToHost, stream()),
          "device_memory_loader::download: after cudaMemcpyAsync"
        );

        return event(device(), stream());
      });
#endif
    }


    // download is customized to avoid creating events in device code
    void download(address_type from, std::size_t num_bytes, void* to) const
    {
#if defined(__CUDACC__)
      if UBU_TARGET(ubu::detail::is_device())
      {
        std::memcpy(to, from, num_bytes);
      }
      else
      {
        wait(download_after(initial_happening(*this), from, num_bytes, to));
      }
#else
      wait(download_after(initial_happening(*this), from, num_bytes, to));
#endif
    }


    event upload_after(const event& before, const void* from, std::size_t num_bytes, address_type to) const
    {
#if defined(__CUDACC__)
      if UBU_TARGET(ubu::detail::is_host())
      {
        return detail::temporarily_with_current_device(device(), [&]
        {
          detail::throw_on_error(cudaStreamWaitEvent(stream(), before.native_handle()),
            "device_memory_loader::upload_after: after cudaStreamWaitEvent"
          );

          detail::throw_on_error(cudaMemcpyAsync(to, from, num_bytes, cudaMemcpyHostToDevice, stream()),
            "device_memory_loader::upload_after: after cudaMemcpyAsync"
          );

          return event(device(), stream());
        });
      }
      else if UBU_TARGET(ubu::detail::is_device())
      {
        // XXX we need to wait on the before event somehow
        std::memcpy(to, from, num_bytes);
        return event(device(), stream());
      }
      else
      {
        // this should never be reached
        assert(0);
        return event(device(), stream());
      }
#else
      return detail::temporarily_with_current_device(device(), [&]
      {
        detail::throw_on_error(cudaStreamWaitEvent(stream(), before.native_handle()),
          "device_memory_loader::upload_after: after cudaStreamWaitEvent"
        );

        detail::throw_on_error(cudaMemcpyAsync(to, from, num_bytes, cudaMemcpyHostToDevice, stream()),
          "device_memory_loader::upload after cudaMemcpyAsync"
        );

        return event(device(), stream());
      });
#endif
    }


    // upload is customized to avoid creating events in device code
    void upload(const void* from, std::size_t num_bytes, address_type to) const
    {
#if defined(__CUDACC__)
      if UBU_TARGET(ubu::detail::is_device())
      {
        std::memcpy(to, from, num_bytes);
      }
      else
      {
        wait(upload_after(initial_happening(*this), from, num_bytes, to));
      }
#else
      wait(upload_after(initial_happening(*this), from, num_bytes, to));
#endif
    }

    constexpr int device() const
    {
      return device_;
    }

    constexpr cudaStream_t stream() const
    {
      return stream_;
    }

    auto operator<=>(const device_memory_loader&) const = default;

  private:
    int device_;
    cudaStream_t stream_;
};


template<plain_old_data_or_void T>
using device_ptr = remote_ptr<T, device_memory_loader>;


} // end ubu::cuda

#include "../../detail/epilogue.hpp"

