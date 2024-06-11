#pragma once

#include "../../detail/prologue.hpp"

#include "initial_happening.hpp"
#include <concepts>
#include <type_traits>


namespace ubu
{


template<class H>
concept happening = 
  std::is_nothrow_move_constructible_v<H>
  and std::is_nothrow_destructible_v<H>

  // a happening must support initial_happening, and its result type must match its argument type
  and requires(std::remove_cvref_t<H>& h)
  {
    { initial_happening(h) } -> std::same_as<std::remove_cvref_t<H>>;
  }
;


} // end ubu


#include "../../detail/epilogue.hpp"

