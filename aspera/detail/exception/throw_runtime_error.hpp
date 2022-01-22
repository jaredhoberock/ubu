#pragma once

#include "../prologue.hpp"

#include "../reflection.hpp"
#include "terminate_with_message.hpp"
#include <stdexcept>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


inline void throw_runtime_error(const char* message)
{
  if ASPERA_TARGET(is_host())
  {
    throw std::runtime_error(message);
  }
  else
  {
    detail::terminate_with_message(message);
  }
}


} // end detail


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"
