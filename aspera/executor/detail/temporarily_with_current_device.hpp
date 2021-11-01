#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/exception.hpp"
#include "../../detail/reflection.hpp"
#include <cuda_runtime_api.h>
#include <functional>
#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


template<std::invocable F>
void temporarily_with_current_device(int device, F&& f)
{
  int old_device = -1;

  if ASPERA_TARGET(has_cuda_runtime())
  {
    detail::throw_on_error(cudaGetDevice(&old_device), "detail::temporarily_with_current_device: CUDA error after cudaGetDevice");
  }
  else
  {
    detail::throw_runtime_error("detail::temporarily_with_current_device: cudaGetDevice is unavailable.");
  }

  if(device != old_device)
  {
    if ASPERA_TARGET(is_device())
    {
      detail::terminate_with_message("detail::temporarily_with_current_device: Requested device cannot differ from current device in __device__ code.");
    }
    else
    {
      detail::throw_on_error(cudaSetDevice(device), "detail::temporarily_with_current_device: CUDA error after cudaSetDevice");
    }
  }

  std::invoke(std::forward<F>(f));

  if(device != old_device)
  {
    if ASPERA_TARGET(is_host())
    {
      detail::throw_on_error(cudaSetDevice(old_device), "detail::temporarily_with_current_device: CUDA error after cudaSetDevice");
    }
  }
};


} // end detail


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

