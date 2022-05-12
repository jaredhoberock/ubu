#pragma once

#include "../detail/prologue.hpp"

#include "../detail/exception/terminate_with_message.hpp"
#include "../detail/exception/throw_runtime_error.hpp"
#include "../detail/for_each_arg.hpp"
#include "../detail/reflection.hpp"
#include "detail/throw_on_cuda_error.hpp"
#include <cuda_runtime_api.h>


ASPERA_NAMESPACE_OPEN_BRACE


namespace cuda
{


// RAII object which always refers to a valid cudaEvent_t resource
class event
{
  public:
    inline event(cudaStream_t s)
      : event{}
    {
      record_on(s);
    }

    inline event(event&& other) noexcept
      : event{0}
    {
      swap(other);
    }

    inline ~event() noexcept
    {
      // XXX should make sure the event is valid before doing any of this

      if(origin_target_ == current_target())
      {
        if ASPERA_TARGET(detail::has_cuda_runtime())
        {
          detail::throw_on_cuda_error(cudaEventDestroy(native_handle()), "cuda::event::~event: after cudaEventDestroy");
        }
        else
        {
          detail::terminate_with_message("cuda::event::~event: cudaEventDestroy is unavailable.");
        }
      }
      else
      {
        printf("Warning: cuda::event::~event: Leaking cudaEvent_t created on different target.");
      }
    }

    inline event& operator=(event&& other)
    {
      swap(other);
      return *this;
    }

    inline cudaEvent_t native_handle() const
    {
      return native_handle_;
    }

    void record_on(cudaStream_t s)
    {
      if ASPERA_TARGET(detail::has_cuda_runtime())
      {
        detail::throw_on_cuda_error(cudaEventRecord(native_handle(), s), "cuda::event::record_on: after cudaEventRecord");
      }
      else
      {
        detail::throw_runtime_error("cuda::event::record_on: cudaEventRecord is unavailable.");
      }
    }

    inline bool is_ready() const
    {
      bool result = false;

      if ASPERA_TARGET(detail::has_cuda_runtime())
      {
        cudaError_t status = cudaEventQuery(native_handle());

        if(status != cudaErrorNotReady and status != cudaSuccess)
        {
          detail::throw_on_cuda_error(status, "cuda::event::is_ready: after cudaEventQuery");
        }

        result = (status == cudaSuccess);
      }
      else
      {
        detail::throw_runtime_error("cuda::event::record_on: cudaEventRecord is unavailable.");
      }

      return result;
    }

    inline void wait() const
    {
      if ASPERA_TARGET(detail::has_cuda_runtime())
      {
        if ASPERA_TARGET(detail::is_device())
        {
          detail::throw_on_cuda_error(cudaDeviceSynchronize(), "cuda::event::wait: after cudaDeviceSynchronize");
        }
        else
        {
          detail::throw_on_cuda_error(cudaEventSynchronize(native_handle()), "cuda::event::wait: after cudaEventSynchronize");
        }
      }
      else
      {
        detail::throw_runtime_error("cuda::event::wait: Unsupported operation.");
      }
    }

    inline static event make_independent_event()
    {
      return {0};
    }

    inline void swap(event& other)
    {
      std::swap(native_handle_, other.native_handle_);
      std::swap(origin_target_, other.origin_target_);
    }

    template<std::same_as<event>... Es>
    event make_dependent_event(const Es&... es) const
    {
      return {0, *this, es...};
    }

  private:
    enum class from {host, device};

    inline static from current_target() noexcept
    {
      from result{from::host};

      if ASPERA_TARGET(detail::is_device())
      {
        result = from::device;
      }

      return result;
    }

    inline static cudaEvent_t make_cuda_event()
    {
      cudaEvent_t result{};

      if ASPERA_TARGET(detail::has_cuda_runtime())
      {
        detail::throw_on_cuda_error(cudaEventCreateWithFlags(&result, cudaEventDisableTiming), "cuda::event::make_cuda_event: after cudaEventCreateWithFlags");
      }
      else
      {
        detail::throw_runtime_error("cuda::event::make_cuda_event: cudaEventCreateWithFlags is unavailable.");
      }

      return result;
    }

    inline event()
      : native_handle_{make_cuda_event()},
        origin_target_{current_target()}
    {}

    // this ctor is available to make_dependent_event
    // the int parameter is simply there to distinguish this ctor from a copy ctor
    template<std::same_as<event>... Es>
    event(int, const event& e, const Es&... es)
      : event{}
    {
      if ASPERA_TARGET(detail::has_cuda_runtime())
      {
        // create a cudaStream_t on which to record our event
        cudaStream_t s{};
        detail::throw_on_cuda_error(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cuda::event ctor: after cudaStreamCreateWithFlags");

        // make the new stream wait on all the event parameters
        detail::for_each_arg([s](const event& e)
        {
          detail::throw_on_cuda_error(cudaStreamWaitEvent(s, e.native_handle()), "cuda::event ctor: after cudaStreamWaitEvent");
        }, e, es...);

        // record our event on the stream
        record_on(s);

        // immediately destroy the stream
        detail::throw_on_cuda_error(cudaStreamDestroy(s), "cuda::event ctor: after cudaStreamDestroy");
      }
      else
      {
        detail::throw_runtime_error("cuda::event ctor: cudaStreamCreateWithFlags is unavailable.");
      }
    }

    cudaEvent_t native_handle_;
    from origin_target_;
};


} // end cuda


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

