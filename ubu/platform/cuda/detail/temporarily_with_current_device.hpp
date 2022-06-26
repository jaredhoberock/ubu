#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../detail/exception/throw_runtime_error.hpp"
#include "has_runtime.hpp"
#include "throw_on_error.hpp"
#include <concepts>
#include <cuda_runtime_api.h>
#include <functional>
#include <utility>


namespace ubu::cuda::detail
{


struct current_device_in_this_scope
{
  int old_device_;
  int new_device_;

  inline current_device_in_this_scope(int new_device)
    : old_device_{-1},
      new_device_{new_device}
  {
    if UBU_TARGET(has_runtime())
    {
      if UBU_TARGET(has_runtime())
      {
        throw_on_error(cudaGetDevice(&old_device_), "cuda::detail::current_device_in_this_scope ctor: CUDA error after cudaGetDevice");
      }
      else
      {
        ubu::detail::throw_runtime_error("cuda::detail::current_device_in_this_scope ctor: cudaGetDevice is unavailable.");
      }
    }

    if(new_device_ != old_device_)
    {
      if UBU_TARGET(ubu::detail::is_device())
      {
        ubu::detail::terminate_with_message("cuda::detail::current_device_in_this_scope ctor:: Requested device cannot differ from current device in __device__ code.");
      }
      else
      {
        throw_on_error(cudaSetDevice(new_device_), "cuda::detail::current_device_in_this_scope ctor: after cudaSetDevice");
      }
    }
  }

  inline ~current_device_in_this_scope()
  {
    if(new_device_ != old_device_)
    {
      if UBU_TARGET(ubu::detail::is_host())
      {
        throw_on_error(cudaSetDevice(old_device_), "cuda::detail::current_device_in_this_scope dtor: after cudaSetDevice");
      }
    }
  }
};


template<std::invocable F>
std::invoke_result_t<F&&> temporarily_with_current_device(int device, F&& f)
{
  current_device_in_this_scope scope{device};

  return std::invoke(std::forward<F>(f));
};


} // end ubu::cuda::detail


#include "../../../detail/epilogue.hpp"

