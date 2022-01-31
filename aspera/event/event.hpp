#pragma once

#include "../detail/prologue.hpp"

#include "wait.hpp"


ASPERA_NAMESPACE_OPEN_BRACE


template<class E>
concept event = requires(std::remove_cvref_t<E> e)
{
  // a lvalue ref to e must be able to wait
  ASPERA_NAMESPACE::wait(e);
};


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

