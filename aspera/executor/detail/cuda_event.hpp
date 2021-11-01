#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/exception.hpp"
#include "../../detail/reflection.hpp"
#include "temporarily_with_current_device.hpp"
#include <cuda_runtime_api.h>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


// RAII object which always refers to a valid cudaEvent_t resource
class cuda_event
{
  public:
    inline cuda_event(int device)
      : device_{device},
        native_handle_{make_cuda_event(device)},
        origin_target_{current_target()}
    {}

    inline cuda_event(int device, cudaStream_t s)
      : cuda_event{device}
    {
      record_on(s);
    }

    inline cuda_event(cuda_event&& other)
      : cuda_event{0}
    {
      swap(other);
    }

    inline ~cuda_event() noexcept
    {
      if(origin_target_ == current_target())
      {
        if ASPERA_TARGET(has_cuda_runtime())
        {
          detail::throw_on_error(cudaEventDestroy(native_handle()), "detail::cuda_event::~cuda_event: CUDA error after cudaEventDestroy");
        }
        else
        {
          detail::terminate_with_message("detail::cuda_event::~cuda_event: cudaEventDestroy is unavailable.");
        }
      }
      else
      {
        printf("Warning: detail::cuda_event::~cuda_event: Leaking cudaEvent_t created on different target.");
      }
    }

    inline cuda_event& operator=(cuda_event&& other)
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
      if ASPERA_TARGET(has_cuda_runtime())
      {
        detail::throw_on_error(cudaEventRecord(native_handle(), s), "detail::cuda_event::record_on: CUDA error after cudaEventRecord");
      }
      else
      {
        detail::throw_runtime_error("detail::cuda_event::record_on: cudaEventRecord is unavailable.");
      }
    }

    inline bool is_ready() const
    {
      bool result = false;

      if ASPERA_TARGET(has_cuda_runtime())
      {
        cudaError_t status = cudaEventQuery(native_handle());

        if(status != cudaErrorNotReady and status != cudaSuccess)
        {
          detail::throw_on_error(status, "detail::cuda_event::is_ready: CUDA error after cudaEventQuery");
        }

        result = (status == cudaSuccess);
      }
      else
      {
        detail::throw_runtime_error("detail::cuda_event::record_on: cudaEventRecord is unavailable.");
      }

      return result;
    }

    inline void wait() const
    {
      if ASPERA_TARGET(has_cuda_runtime())
      {
        if ASPERA_TARGET(is_device())
        {
          detail::throw_on_error(cudaDeviceSynchronize(), "detail::cuda_event::wait: CUDA error after cudaDeviceSynchronize");
        }
        else
        {
          detail::throw_on_error(cudaEventSynchronize(native_handle()), "detail::cuda_event::wait: CUDA error after cudaEventSynchronize");
        }
      }
      else
      {
        detail::throw_runtime_error("detail::cuda_event::wait: Unsupported operation.");
      }
    }

    inline void swap(cuda_event& other)
    {
      std::swap(device_, other.device_);
      std::swap(native_handle_, other.native_handle_);
      std::swap(origin_target_, other.origin_target_);
    }

  private:
    enum class from {host, device};

    inline static from current_target() noexcept
    {
      from result{from::host};

      if ASPERA_TARGET(is_device())
      {
        result = from::device;
      }

      return result;
    }

    inline static cudaEvent_t make_cuda_event(int device)
    {
      cudaEvent_t result{};

      detail::temporarily_with_current_device(device, [&result]
      {
        if ASPERA_TARGET(has_cuda_runtime())
        {
          detail::throw_on_error(cudaEventCreateWithFlags(&result, cudaEventDisableTiming), "detail::cuda_event::make_cuda_event: CUDA error after cudaEventCreateWithFlags");
        }
        else
        {
          detail::throw_runtime_error("detail::cuda_event::make_cuda_event: cudaEventCreateWithFlags is unavailable.");
        }
      });

      return result;
    }

    int device_;
    cudaEvent_t native_handle_;
    from origin_target_;
};


} // end detail


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../../detail/epilogue.hpp"

