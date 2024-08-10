#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../tensors/concepts/decomposable.hpp"
#include "../../../tensors/concepts/view_of.hpp"
#include "../../../tensors/coordinates/concepts/coordinate.hpp"
#include "../../../tensors/views/layouts/concepts/coshaped_layout.hpp"
#include "../../../tensors/vectors/span_like.hpp"
#include "../../../utilities/tuples.hpp"
#include <type_traits>

namespace ubu
{
namespace detail
{


template<class T>
constexpr bool is_memory_view()
{
  // T is a memory_view if:
  // 1. T is span_like, or
  // 2. T can be decomposed into a (memory_view, coshaped_layout)

  if constexpr (span_like<T>)
  {
    return true;
  }
  else if constexpr (decomposable<T>)
  {
    using L = tuples::second_t<T>;

    if constexpr (coshaped_layout<L>)
    {
      using V = tuples::first_t<T>;

      return is_memory_view<V>();
    }
    else
    {
      return false;
    }
  }
  else
  {
    return false;
  }
}


} // end detail

template<class T>
concept memory_view = view<T> and detail::is_memory_view<T>();

template<class T, class E, class S = void>
concept memory_view_of = memory_view<T> and view_of<T,E,S>;

} // end ubu

#include "../../../detail/epilogue.hpp"

