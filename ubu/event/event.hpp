#pragma once

#include "../detail/prologue.hpp"

#include "make_dependent_event.hpp"
#include "wait.hpp"
#include <concepts>
#include <utility>


UBU_NAMESPACE_OPEN_BRACE


template<class E>
concept event = 
  std::is_nothrow_move_constructible_v<E>
  and std::is_nothrow_destructible_v<E>

  // a mutable ref to e must be able to wait
  and requires(std::remove_cvref_t<E>& e)
  {
    UBU_NAMESPACE::wait(e);
  }

  // a dependent event must be constructible from two rvalue refs
  and requires(std::remove_cvref_t<E>& e1, std::remove_cvref_t<E>& e2)
  {
    { UBU_NAMESPACE::make_dependent_event(std::move(e1), std::move(e2)) } -> std::same_as<std::remove_cvref_t<E>>;
  }
;


UBU_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

