#pragma once

#include "../../detail/prologue.hpp"

#include "detail/throw_on_error.hpp"
#include "event.hpp"
#include <cuda_runtime_api.h>


namespace ubu::cuda
{


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

    using happening_type = event;

    template<std::invocable F>
    event execute_after(const event& before, F&& f) const noexcept
    {
      using T = std::decay_t<F>;

      // allocate a copy of f
      // XXX try to use an allocator instead of new
      // XXX in practice, some execution context should manage this lifetime
      T* ptr_to_f = new T{std::forward<F>(f)};

      // make the stream wait for the before event
      detail::throw_on_error(cudaStreamWaitEvent(stream(), before.native_handle()), "callback_executor::execute_after: CUDA error after cudaStreamWaitEvent");

      // enqueue the callback
      detail::throw_on_error(cudaStreamAddCallback(stream_, &callback<T>, ptr_to_f, 0), "callback_executor::execute_after: CUDA error after cudaStreamAddCallback");

      // return a new event recorded on our stream
      return event{0, stream_};
    }

    auto operator<=>(const callback_executor&) const = default;

    inline cudaStream_t stream() const noexcept
    {
      return stream_;
    }
};


} // end namespace ubu::cuda


#include "../../detail/epilogue.hpp"

