#pragma once

#include "../detail/prologue.hpp"

#include "detail/has_runtime.hpp"
#include <cuda_runtime.h>
#include <system_error>

UBU_NAMESPACE_OPEN_BRACE


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
      if UBU_TARGET(detail::has_runtime())
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


UBU_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

