#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../tensors/concepts/decomposable.hpp"
#include "../../../tensors/concepts/nested_tensor_like.hpp"
#include "../../../tensors/concepts/tensor_like_of.hpp"
#include "../../../tensors/coordinates/concepts/congruent.hpp"
#include "../../../tensors/coordinates/concepts/coordinate.hpp"
#include "../../../tensors/views/layouts/concepts/coshaped_layout.hpp"
#include "../../../tensors/vectors/span_like.hpp"
#include "../../../utilities/tuples.hpp"
#include <type_traits>

namespace ubu
{
namespace detail
{


template<class T, class E = void>
constexpr bool is_memory_view_of()
{
  // T is a memory view of E if T is a view and:
  // 1. T is a span of E, or
  // 2. T is nested and T's elements are memory views of E
  // 3. T can be decomposed into a (memory_view_of<E>, coshaped_layout)

  if constexpr (view<T>)
  {
    if constexpr (span_like<T> and (std::is_void_v<E> or tensor_like_of<T,E>))
    {
      return true;
    }
    else if constexpr (nested_tensor_like<T> and is_memory_view_of<tensor_element_t<T>,E>())
    {
      return true;
    }
    else if constexpr (decomposable<T>)
    {
      using Pair = decompose_result_t<T>;

      using L = tuples::second_t<Pair>;

      if constexpr (coshaped_layout<L>)
      {
        using U = tuples::first_t<Pair>;

        return is_memory_view_of<U,E>();
      }
    }
  }

  return false;
}


} // end detail


template<class T, class E = void, class S = void>
concept memory_view_of =
  detail::is_memory_view_of<T,E>()
  and (std::is_void_v<S> or (coordinate<S> and congruent<S,tensor_shape_t<T>>))
;

template<class T>
concept memory_view = memory_view_of<T>;

} // end ubu

#include "../../../detail/epilogue.hpp"

