#pragma once

#include "../detail/prologue.hpp"

#include "detail/cuda_event.hpp"
#include <cuda_runtime_api.h>


ASPERA_NAMESPACE_OPEN_BRACE


class callback_executor
{
  private:
    cudaStream_t stream_;

    template<std::invocable F>
    static void callback(cudaStream_t stream, cudaError_t status, void* data)
    {
      F* func = reinterpret_cast<F*>(data);

      if(status == cudaSuccess)
      {
        // call the function
        (*func)();
      }
      else
      {
        // report the error somehow
        // ...
      }

      // delete the function
      // XXX try to use an allocator instead of delete
      delete func;
    }

  public:
    inline explicit callback_executor(cudaStream_t stream)
      : stream_{stream}
    {}

    callback_executor(const callback_executor&) = default;

    template<std::invocable F>
    detail::cuda_event execute(F&& f) const noexcept
    {
      using T = std::decay_t<F>;

      // allocate a copy of f
      // XXX try to use an allocator instead of new
      // XXX in practice, some execution context should manage this lifetime
      T* ptr_to_f = new T{std::forward<F>(f)};

      // enqueue the callback
      detail::throw_on_error(cudaStreamAddCallback(stream_, &callback<T>, ptr_to_f, 0), "callback_executor::execute: CUDA error after cudaStreamAddCallback");

      // return a new event recorded on our stream
      return detail::cuda_event{0, stream_};
    }

    auto operator<=>(const callback_executor&) const = default;

    inline cudaStream_t stream() const noexcept
    {
      return stream_;
    }

    //inline blocking_t query(blocking_t) const noexcept
    //{
    //  return blocking.possibly;
    //}
};

ASPERA_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

