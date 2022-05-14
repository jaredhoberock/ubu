#pragma once

#include "../detail/prologue.hpp"

#include "wait.hpp"


ASPERA_NAMESPACE_OPEN_BRACE


template<class E>
concept event = 
  std::is_nothrow_move_constructible_v<E> and
  std::is_nothrow_destructible_v<E> and
  requires(std::remove_cvref_t<E>& e)
  {
    // a mutable ref to e must be able to wait
    ASPERA_NAMESPACE::wait(e);
  }
;


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

