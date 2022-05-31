#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/exception/throw_on_error.hpp"
#include "../error_category.hpp"

UBU_NAMESPACE_OPEN_BRACE

namespace detail
{


inline int throw_on_cl_error(cl_int e, const char* message)
{
  return detail::throw_on_error(e, cl::error_category(), message);
}


} // end detail


UBU_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

