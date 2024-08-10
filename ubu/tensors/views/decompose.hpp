#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/tuples.hpp"
#include "../concepts/composable.hpp"
#include "../concepts/view.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

template<class P>
concept composable_pair =
  tuples::pair_like<P>
  and composable<tuples::first_t<P>,tuples::second_t<P>>
;

template<class T>
concept has_decompose_member_function = requires(T tensor)
{
  { tensor.decompose() } -> composable_pair;

  // XXX should also require that the element type of the A and the element type of T are the same
  // XXX should also require that the shape of B and the shape of T are congruent
};

template<class T>
concept has_decompose_free_function = requires(T tensor)
{
  { decompose(tensor) } -> composable_pair;

  // XXX should also require that the element type of the A and the element type of T are the same
  // XXX should also require that the shape of B and the shape of T are congruent
};


struct dispatch_decompose
{
  template<class T>
    requires has_decompose_member_function<T&&>
  constexpr composable_pair auto operator()(T&& tensor) const
  {
    return std::forward<T>(tensor).decompose();
  }

  template<class T>
    requires (not has_decompose_member_function<T&&>
              and has_decompose_free_function<T&&>)
  constexpr composable_pair auto operator()(T&& tensor) const
  {
    return decompose(std::forward<T>(tensor));
  }
};


} // end detail

inline constexpr detail::dispatch_decompose decompose;

template<class T>
using decompose_result_t = decltype(decompose(std::declval<T>()));

} // end ubu

#include "../../detail/epilogue.hpp"

