#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/tuples.hpp"
#include "../memory/views/memory_view.hpp"
#include "happening.hpp"

namespace ubu
{

template<class T, class E, class S = void>
concept asynchronous_memory_view_of =
  tuples::pair_like<T>
  and happening<tuples::first_t<T>>
  and memory_view_of<tuples::second_t<T>, E, S>
;

} // end ubu

#include "../../detail/epilogue.hpp"

