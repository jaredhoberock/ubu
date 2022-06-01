#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/exception/throw_on_error.hpp"
#include "../error_category.hpp"


namespace ubu::detail
{


inline int throw_on_cuda_error(cudaError_t e, const char* message)
{
  return detail::throw_on_error(e, cuda::error_category(), message);
}


} // end ubu::detail


#include "../../detail/epilogue.hpp"

