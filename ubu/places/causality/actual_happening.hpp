#pragma once

#include "../../detail/prologue.hpp"

#include "happening.hpp"
#include "has_happened.hpp"
#include <type_traits>


namespace ubu
{


template<class H>
concept actual_happening =
  happening<H>

  // an actual_happening must report whether it has happened or not
  and requires(const std::remove_cvref_t<H>& h)
  {
    ubu::has_happened(h);
  }
;


} // end ubu


#include "../../detail/epilogue.hpp"

