#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/tuples.hpp"
#include "../concepts/composable.hpp"
#include "../concepts/view.hpp"
#include "../concepts/viewable_tensor_like.hpp"
#include "all.hpp"
#include "detail/view_of_composition.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

template<class P, class C>
concept decomposition_of =
  tuples::pair_like<P>
  and composable<tuples::first_t<P>,tuples::second_t<P>>
  and viewable_tensor_like<C>
  and view_of_composition<all_t<C>,tuples::first_t<P>,tuples::second_t<P>>
;

template<class T>
concept has_decompose_member_function = requires(T tensor)
{
  { tensor.decompose() } -> decomposition_of<T>;
};

template<class T>
concept has_decompose_free_function = requires(T tensor)
{
  { decompose(tensor) } -> decomposition_of<T>;
};


struct dispatch_decompose
{
  template<class T>
    requires has_decompose_member_function<T&&>
  constexpr decomposition_of<T&&> auto operator()(T&& tensor) const
  {
    return std::forward<T>(tensor).decompose();
  }

  template<class T>
    requires (not has_decompose_member_function<T&&>
              and has_decompose_free_function<T&&>)
  constexpr decomposition_of<T&&> auto operator()(T&& tensor) const
  {
    return decompose(std::forward<T>(tensor));
  }
};


} // end detail

inline constexpr detail::dispatch_decompose decompose;

template<class T>
using decompose_result_t = decltype(decompose(std::declval<T>()));

template<class T>
using decompose_result_first_t = tuples::first_t<decompose_result_t<T>>;

template<class T>
using decompose_result_second_t = tuples::second_t<decompose_result_t<T>>;

} // end ubu

#include "../../detail/epilogue.hpp"

