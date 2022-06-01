#pragma once

#include "../../detail/prologue.hpp"

#include "associated_executor.hpp"


namespace ubu
{


template<class T>
concept executor_associate = requires(T arg)
{
  associated_executor(arg);
};


} // end ubu

#include "../../detail/epilogue.hpp"

