#pragma once

#include "../detail/prologue.hpp"

#include "associated_executor.hpp"


ASPERA_NAMESPACE_OPEN_BRACE


template<class T>
concept executor_associate = requires(T arg)
{
  ASPERA_NAMESPACE::associated_executor(arg);
};


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

