#pragma once

#include "../detail/prologue.hpp"

#include "because_of.hpp"
#include "wait.hpp"
#include <concepts>
#include <utility>


namespace ubu
{


template<class H>
concept happening = 
  std::is_nothrow_move_constructible_v<H>
  and std::is_nothrow_destructible_v<H>

  // a mutable ref to h must be able to wait
  and requires(std::remove_cvref_t<H>& h)
  {
    ubu::wait(h);
  }

  // a happening must be the effect of two lvalue refs
  and requires(std::remove_cvref_t<H>& h1, std::remove_cvref_t<H>& h2)
  {
    { ubu::because_of(std::move(h1), std::move(h2)) } -> std::same_as<std::remove_cvref_t<H>>;
  }
;


} // end ubu


#include "../detail/epilogue.hpp"

