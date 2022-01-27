#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/exception.hpp"
#include "../../detail/reflection.hpp"
#include <concepts>
#include <cuda_runtime_api.h>
#include <functional>
#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


struct current_cuda_device_in_this_scope
{
  int old_device_;
  int new_device_;

  inline current_cuda_device_in_this_scope(int new_device)
    : old_device_{-1},
      new_device_{new_device}
  {
    if ASPERA_TARGET(has_cuda_runtime())
    {
      if ASPERA_TARGET(has_cuda_runtime())
      {
        detail::throw_on_error(cudaGetDevice(&old_device_), "detail::current_cuda_device_in_this_scope ctor: CUDA error after cudaGetDevice");
      }
      else
      {
        detail::throw_runtime_error("detail::current_cuda_device_in_this_scope ctor: cudaGetDevice is unavailable.");
      }
    }

    if(new_device_ != old_device_)
    {
      if ASPERA_TARGET(is_device())
      {
        detail::terminate_with_message("detail::current_cuda_device_in_this_scope ctor:: Requested device cannot differ from current device in __device__ code.");
      }
      else
      {
        detail::throw_on_error(cudaSetDevice(old_device_), "detail::current_cuda_device_in_this_scope ctor: CUDA error after cudaSetDevice");
      }
    }
  }

  inline ~current_cuda_device_in_this_scope()
  {
    if(new_device_ != old_device_)
    {
      if ASPERA_TARGET(is_host())
      {
        detail::throw_on_error(cudaSetDevice(old_device_), "detail::current_cuda_device_in_this_scope dtor: CUDA error after cudaSetDevice");
      }
    }
  }
};


template<std::invocable F>
std::invoke_result_t<F&&> temporarily_with_current_device(int device, F&& f)
{
  current_cuda_device_in_this_scope scope{device};

  return std::invoke(std::forward<F>(f));
};


} // end detail


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"
