#pragma once

#include "../detail/prologue.hpp"

#include "../detail/reflection.hpp"
#include <cuda_runtime.h>
#include <system_error>

ASPERA_NAMESPACE_OPEN_BRACE


namespace cuda
{


class error_category : public std::error_category
{
  public:
    inline const char* name() const noexcept
    {
      return "CUDA Runtime";
    }

    inline std::string message(int ev) const
    {
      if ASPERA_TARGET(detail::has_cuda_runtime())
      {
        return cudaGetErrorString(static_cast<cudaError_t>(ev));
      }
      else
      {
        return "Unknown error";
      }
    }
};


} // end cuda


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

