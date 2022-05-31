#pragma once

#include "../../detail/prologue.hpp"

#include "associated_executor.hpp"


UBU_NAMESPACE_OPEN_BRACE


template<class T>
concept executor_associate = requires(T arg)
{
  associated_executor(arg);
};


UBU_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

