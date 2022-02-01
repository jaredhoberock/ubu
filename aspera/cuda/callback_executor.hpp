#pragma once

#include "../detail/prologue.hpp"

#include "event.hpp"
#include <cuda_runtime_api.h>


ASPERA_NAMESPACE_OPEN_BRACE


namespace cuda
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

    template<std::invocable F>
    void execute(F&& f) const noexcept
    {
      using T = std::decay_t<F>;

      // allocate a copy of f
      // XXX try to use an allocator instead of new
      // XXX in practice, some execution context should manage this lifetime
      T* ptr_to_f = new T{std::forward<F>(f)};

      // enqueue the callback
      detail::throw_on_error(cudaStreamAddCallback(stream_, &callback<T>, ptr_to_f, 0), "callback_executor::execute: CUDA error after cudaStreamAddCallback");
    }

    template<std::invocable F>
    event first_execute(F&& f) const noexcept
    {
      // execute f
      execute(std::forward<F>(f));

      // return a new event recorded on our stream
      return event{0, stream_};
    }

    template<std::invocable F>
    event execute_after(const event& before, F&& f) const noexcept
    {
      // make the stream wait for the before event
      detail::throw_on_error(cudaStreamWaitEvent(stream(), before.native_handle()), "callback_executor::execute_after: CUDA error after cudaStreamWaitEvent");

      // execute f and return the event
      return first_execute(std::forward<F>(f));
    }

    template<std::invocable F>
    void finally_execute(const event& before, F&& f) const noexcept
    {
      // make the stream wait for the before event
      detail::throw_on_error(cudaStreamWaitEvent(stream(), before.native_handle()), "callback_executor::execute_after: CUDA error after cudaStreamWaitEvent");

      // execute f
      execute(std::forward<F>(f));
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


} // end namespace cuda


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

