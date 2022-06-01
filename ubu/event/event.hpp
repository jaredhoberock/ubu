#pragma once

#include "../detail/prologue.hpp"

#include "make_dependent_event.hpp"
#include "wait.hpp"
#include <concepts>
#include <utility>


namespace ubu
{


template<class E>
concept event = 
  std::is_nothrow_move_constructible_v<E>
  and std::is_nothrow_destructible_v<E>

  // a mutable ref to e must be able to wait
  and requires(std::remove_cvref_t<E>& e)
  {
    ubu::wait(e);
  }

  // a dependent event must be constructible from two rvalue refs
  and requires(std::remove_cvref_t<E>& e1, std::remove_cvref_t<E>& e2)
  {
    { ubu::make_dependent_event(std::move(e1), std::move(e2)) } -> std::same_as<std::remove_cvref_t<E>>;
  }
;


} // end ubu


#include "../detail/epilogue.hpp"

