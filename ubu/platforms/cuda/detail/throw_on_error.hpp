#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../detail/exception/throw_on_error.hpp"
#include "../error_category.hpp"


namespace ubu::cuda::detail
{


inline int throw_on_error(cudaError_t e, const char* message)
{
  return ubu::detail::throw_on_error(e, cuda::error_category(), message);
}


}


#include "../../../detail/epilogue.hpp"

