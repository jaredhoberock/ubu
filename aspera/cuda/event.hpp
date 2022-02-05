#pragma once

#include "../detail/prologue.hpp"

#include "../detail/exception.hpp"
#include "../detail/reflection.hpp"
#include <cuda_runtime_api.h>


ASPERA_NAMESPACE_OPEN_BRACE


namespace cuda
{


// RAII object which always refers to a valid cudaEvent_t resource
class event
{
  public:
    inline event()
      : native_handle_{make_cuda_event()},
        origin_target_{current_target()}
    {}

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
          detail::throw_on_error(cudaEventDestroy(native_handle()), "cuda::event::~event: CUDA error after cudaEventDestroy");
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
        detail::throw_on_error(cudaEventRecord(native_handle(), s), "cuda::event::record_on: CUDA error after cudaEventRecord");
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
          detail::throw_on_error(status, "cuda::event::is_ready: CUDA error after cudaEventQuery");
        }

        result = (status == cudaSuccess);
      }
      else
      {
        detail::throw_runtime_error("cuda::event::record_on: cudaEventRecord is unavailable.");
      }

      return result;
    }

    inline void wait()
    {
      if ASPERA_TARGET(detail::has_cuda_runtime())
      {
        if ASPERA_TARGET(detail::is_device())
        {
          detail::throw_on_error(cudaDeviceSynchronize(), "cuda::event::wait: CUDA error after cudaDeviceSynchronize");
        }
        else
        {
          detail::throw_on_error(cudaEventSynchronize(native_handle()), "cuda::event::wait: CUDA error after cudaEventSynchronize");
        }
      }
      else
      {
        detail::throw_runtime_error("cuda::event::wait: Unsupported operation.");
      }
    }

    inline static event make_complete_event()
    {
      return {0};
    }

    inline void swap(event& other)
    {
      std::swap(native_handle_, other.native_handle_);
      std::swap(origin_target_, other.origin_target_);
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
        detail::throw_on_error(cudaEventCreateWithFlags(&result, cudaEventDisableTiming), "cuda::event::make_cuda_event: CUDA error after cudaEventCreateWithFlags");
      }
      else
      {
        detail::throw_runtime_error("cuda::event::make_cuda_event: cudaEventCreateWithFlags is unavailable.");
      }

      return result;
    }

    cudaEvent_t native_handle_;
    from origin_target_;
};


} // end cuda


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

