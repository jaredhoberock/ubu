#pragma once

#include "../prologue.hpp"

#include "../reflection.hpp"
#include "terminate_with_message.hpp"
#include <stdexcept>


namespace ubu::detail
{


inline void throw_runtime_error(const char* message)
{
  if UBU_TARGET(is_host())
  {
    throw std::runtime_error(message);
  }
  else
  {
    detail::terminate_with_message(message);
  }
}


} // end ubu::detail


#include "../epilogue.hpp"

