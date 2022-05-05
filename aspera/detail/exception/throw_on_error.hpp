#pragma once

#include "../prologue.hpp"

#include "../reflection/has_exceptions.hpp"
#include "terminate.hpp"
#include <cstdio>
#include <system_error>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


// this function returns a value so that it can be called on parameter packs
inline int throw_on_error(int e, const std::error_category& category, const char* message)
{
  if(e)
  {
    if ASPERA_TARGET(has_exceptions())
    {
      throw std::system_error(e, category, message);
    }
    else
    {
      printf("%s: %s error [%s]\n", message, category.name(), category.message(e).c_str());
      detail::terminate();
    }
  }

  return 0;
}


} // end detail


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

