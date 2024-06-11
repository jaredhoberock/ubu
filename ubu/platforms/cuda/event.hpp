#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/exception/terminate_with_message.hpp"
#include "../../detail/exception/throw_runtime_error.hpp"
#include "../../detail/for_each_arg.hpp"
#include "detail/has_runtime.hpp"
#include "detail/temporarily_with_current_device.hpp"
#include "detail/throw_on_error.hpp"
#include <cuda_runtime_api.h>


namespace ubu::cuda
{


// RAII object which always refers to a valid cudaEvent_t resource
class event
{
  public:
    inline event(int device, cudaStream_t s)
      : event{device}
    {
      record_on(s);
    }

    inline event(event&& other) noexcept
      : device_{-1},
        native_handle_{0},
        origin_target_{current_target()}
    {
      swap(other);
    }

    inline ~event() noexcept
    {
      if(native_handle_)
      {
        if(origin_target_ == current_target())
        {
          if UBU_TARGET(detail::has_runtime())
          {
            detail::throw_on_error(cudaEventDestroy(native_handle()), "cuda::event::~event: after cudaEventDestroy");
          }
          else
          {
            ubu::detail::terminate_with_message("cuda::event::~event: cudaEventDestroy is unavailable.");
          }
        }
        else
        {
          printf("Warning: cuda::event::~event: Leaking cudaEvent_t created on different target.");
        }
      }
    }

    inline event& operator=(event&& other)
    {
      swap(other);
      return *this;
    }

    inline int device() const
    {
      return device_;
    }

    inline cudaEvent_t native_handle() const
    {
      return native_handle_;
    }

    void record_on(cudaStream_t s)
    {
      if UBU_TARGET(detail::has_runtime())
      {
        detail::temporarily_with_current_device(device(), [&]
        {
          detail::throw_on_error(cudaEventRecord(native_handle(), s), "cuda::event::record_on: after cudaEventRecord");
        });
      }
      else
      {
        ubu::detail::throw_runtime_error("cuda::event::record_on: cudaEventRecord is unavailable.");
      }
    }

    inline bool has_happened() const
    {
      bool result = false;

      if UBU_TARGET(detail::has_runtime())
      {
        cudaError_t status = cudaEventQuery(native_handle());

        if(status != cudaErrorNotReady and status != cudaSuccess)
        {
          detail::throw_on_error(status, "cuda::event::has_happened: after cudaEventQuery");
        }

        result = (status == cudaSuccess);
      }
      else
      {
        ubu::detail::throw_runtime_error("cuda::event::record_on: cudaEventRecord is unavailable.");
      }

      return result;
    }

    inline void wait() const
    {
      if UBU_TARGET(detail::has_runtime())
      {
        if UBU_TARGET(ubu::detail::is_device())
        {
          detail::throw_on_error(cudaDeviceSynchronize(), "cuda::event::wait: after cudaDeviceSynchronize");
        }
        else
        {
          detail::throw_on_error(cudaEventSynchronize(native_handle()), "cuda::event::wait: after cudaEventSynchronize");
        }
      }
      else
      {
        ubu::detail::throw_runtime_error("cuda::event::wait: Unsupported operation.");
      }
    }

    inline static event initial_happening()
    {
      return {0, cudaStream_t{0}};
    }

    inline void swap(event& other)
    {
      std::swap(device_, other.device_);
      std::swap(native_handle_, other.native_handle_);
      std::swap(origin_target_, other.origin_target_);
    }

    template<std::same_as<event>... Es>
    event after_all(const Es&... es) const
    {
      return {device_, *this, es...};
    }

  private:
    enum class from {host, device};

    inline static from current_target() noexcept
    {
      from result{from::host};

      if UBU_TARGET(ubu::detail::is_device())
      {
        result = from::device;
      }

      return result;
    }

    inline static cudaEvent_t make_cuda_event(int device)
    {
      cudaEvent_t result{};

      if UBU_TARGET(detail::has_runtime())
      {
        detail::temporarily_with_current_device(device, [&]
        {
          detail::throw_on_error(cudaEventCreateWithFlags(&result, cudaEventDisableTiming), "cuda::event::make_cuda_event: after cudaEventCreateWithFlags");
        });
      }
      else
      {
        ubu::detail::throw_runtime_error("cuda::event::make_cuda_event: cudaEventCreateWithFlags is unavailable.");
      }

      return result;
    }

    inline event(int device)
      : device_{device},
        native_handle_{make_cuda_event(device)},
        origin_target_{current_target()}
    {}

    // this ctor is available to after_all
    template<std::same_as<event>... Es>
    event(int device, const event& e, const Es&... es)
      : event{device}
    {
      if UBU_TARGET(detail::has_runtime())
      {
        detail::temporarily_with_current_device(device, [&]
        {
          // create a cudaStream_t on which to record our event
          cudaStream_t s{};
          detail::throw_on_error(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cuda::event ctor: after cudaStreamCreateWithFlags");

          // make the new stream wait on all the event parameters
          ubu::detail::for_each_arg([s](const event& e)
          {
            detail::throw_on_error(cudaStreamWaitEvent(s, e.native_handle()), "cuda::event ctor: after cudaStreamWaitEvent");
          }, e, es...);

          // record our event on the stream
          record_on(s);

          // immediately destroy the stream
          detail::throw_on_error(cudaStreamDestroy(s), "cuda::event ctor: after cudaStreamDestroy");
        });
      }
      else
      {
        ubu::detail::throw_runtime_error("cuda::event ctor: cudaStreamCreateWithFlags is unavailable.");
      }
    }

    int device_;
    cudaEvent_t native_handle_;
    from origin_target_;
};


} // end ubu::cuda


#include "../../detail/epilogue.hpp"

