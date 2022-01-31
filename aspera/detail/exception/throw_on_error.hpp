#pragma once

#include "../prologue.hpp"

#include "../reflection.hpp"
#include "terminate.hpp"
#include <cuda_runtime_api.h>
#include <stdexcept>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


// this function returns a value so that it can be called on parameter packs
inline int throw_on_error(cudaError_t e, const char* message)
{
  if(e)
  {
    if ASPERA_TARGET(is_host())
    {
      std::string what = std::string(message) + std::string(": ") + cudaGetErrorString(e);
      throw std::runtime_error(what);
    }
    else
    {
      if ASPERA_TARGET(has_cuda_runtime())
      {
        message = cudaGetErrorString(e);

        printf("Error after %s: %s\n", message, cudaGetErrorString(e));
      }
      else
      {
        printf("Error: %s\n", message);
      }

      terminate();
    }
  }

  return 0;
}


} // end detail


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

