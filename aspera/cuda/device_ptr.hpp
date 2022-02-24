#pragma once

#include "../detail/prologue.hpp"

#include "../detail/exception.hpp"
#include "../detail/reflection.hpp"
#include "../memory/fancy_ptr.hpp"
#include "detail/temporarily_with_current_device.hpp"
#include "event.hpp"
#include "kernel_executor.hpp"
#include <cassert>
#include <concepts>
#include <cstring>
#include <type_traits>


ASPERA_NAMESPACE_OPEN_BRACE


namespace cuda
{


template<class T>
  requires (std::is_standard_layout_v<T> and std::is_trivial_v<T>)
class device_memory_copier
{
  public:
    using handle_type = T*;
    using element_type = T;
    using value_type = std::remove_cv_t<T>;

    constexpr device_memory_copier(int device)
      : device_{device}
    {}

    constexpr device_memory_copier()
      : device_memory_copier{0}
    {}

    device_memory_copier(const device_memory_copier&) = default;

    value_type* copy_n(const value_type* from, std::size_t count, value_type* to) const
    {
#if defined(__CUDACC__)
      if ASPERA_TARGET(detail::is_host())
      {
        detail::temporarily_with_current_device(device_, [=]
        {
          detail::throw_on_error(cudaMemcpy(to, from, sizeof(value_type) * count, cudaMemcpyDefault),
            "device_memory_copier: CUDA error after cudaMemcpy"
          );
        });
      }
      else if ASPERA_TARGET(detail::is_device())
      {
        std::memcpy(to, from, sizeof(value_type) * count);
      }
      else
      {
        // this should never be reached
        assert(0);
      }
#else
      detail::temporarily_with_current_device(device_, [=]
      {
        detail::throw_on_error(cudaMemcpy(to, from, sizeof(value_type) * count, cudaMemcpyDefault),
          "device_memory_copier: CUDA error after cudaMemcpy"
        );
      });
#endif

      return to + count;
    }

    value_type* copy_n_to_raw_pointer(const value_type* from, std::size_t count, value_type* to) const
    {
      return this->copy_n(from, count, to);
    }

    value_type* copy_n_from_raw_pointer(const value_type* from, std::size_t count, value_type* to) const
    {
      return this->copy_n(from, count, to);
    }

    constexpr int device() const
    {
      return device_;
    }

  private:
    int device_;
};


template<class T>
using device_ptr = fancy_ptr<T, device_memory_copier<T>>;


// copy_n_after overloads

template<class T, class U>
  requires (std::is_trivially_assignable_v<T&,U> and std::same_as<U,std::remove_cv_t<T>>)
event copy_n_after(kernel_executor ex, event&& before, device_ptr<T> from, std::size_t count, device_ptr<U> to)
{
  // make the stream wait for the before event
  detail::throw_on_error(cudaStreamWaitEvent(ex.stream(), before.native_handle()),
    "cuda::copy_n_after: CUDA error after cudaStreamWaitEvent"
  );

  // enqueue a cudaMemcpyAsync
  detail::throw_on_error(cudaMemcpyAsync(to.native_handle(), from.native_handle(), sizeof(U) * count, cudaMemcpyDeviceToDevice, ex.stream()),
    "cuda::copy_n_after: CUDA error after cudaMemcpyAsync"
  );

  // reuse the input event
  before.record_on(ex.stream());
  
  return std::move(before);
}


template<class T, class U>
  requires (std::is_trivially_assignable_v<T&,U> and std::same_as<U,std::remove_cv_t<T>>)
event copy_n_after(kernel_executor ex, const event& before, device_ptr<T> from, std::size_t count, device_ptr<U> to)
{
  // make the stream wait for the before event
  detail::throw_on_error(cudaStreamWaitEvent(ex.stream(), before.native_handle()),
    "cuda::copy_n_after: CUDA error after cudaStreamWaitEvent"
  );

  // enqueue a cudaMemcpyAsync
  detail::throw_on_error(cudaMemcpyAsync(to.native_handle(), from.native_handle(), sizeof(U) * count, cudaMemcpyDeviceToDevice, ex.stream()),
    "cuda::copy_n_after: CUDA error after cudaMemcpyAsync"
  );

  // create a new event
  return event{ex.stream()};
}


} // end cuda


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

