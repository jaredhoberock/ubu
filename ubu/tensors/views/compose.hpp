#pragma once

#include "../../detail/prologue.hpp"

#include "../concepts/composable.hpp"
#include "../concepts/decomposable.hpp"
#include "../concepts/tensor_like.hpp"
#include "../concepts/viewable_tensor_like.hpp"
#include "all.hpp"
#include "composed_view.hpp"
#include "decompose.hpp"
#include "detail/view_of_composition.hpp"
#include "composed_view.hpp"
#include <type_traits>
#include <utility>

namespace ubu
{
namespace detail
{


// compose and composed_view have a cyclic dependency (via composed_tensor) and can't use each other directly
// declare detail::make_composed_view for compose's use
template<class A, view B>
  requires (std::is_trivially_copy_constructible_v<A> and composable<A,B> and (view<A> or not tensor_like<A>))
constexpr view auto make_composed_view(A, B);

template<class T>
concept viewable_tensor_like_or_not_tensor_like =
  viewable_tensor_like<T> or not tensor_like<T>
;

template<class A, class B>
concept has_compose_member_function = requires(A a, B b)
{
  { a.compose(b) } -> view_of_composition<A,B>;
};

template<class A, class B>
concept has_compose_free_function = requires(A a, B b)
{
  { compose(a,b) } -> view_of_composition<A,B>;
};

struct dispatch_compose
{
  template<class A, class B>
    requires has_compose_member_function<A&&,B&&>
  constexpr view_of_composition<A&&,B&&> auto operator()(A&& a, B&& b) const
  {
    return std::forward<A>(a).compose(std::forward<B>(b));
  }

  template<class A, class B>
    requires (not has_compose_member_function<A&&,B&&>
              and has_compose_free_function<A&&,B&&>)
  constexpr view_of_composition<A&&,B&&> auto operator()(A&& a, B&& b) const
  {
    return compose(std::forward<A>(a), std::forward<B>(b));
  }

  template<viewable_tensor_like_or_not_tensor_like A, tensor_like B>
    requires (not has_compose_member_function<A&&,B&&>
              and not has_compose_free_function<A&&,B&&>
              and composable<A,B>)
  constexpr view_of_composition<A&&,B&&> auto operator()(A&& a, B&& b) const
  {
    if constexpr (decomposable<A&&>)
    {
      // decompose a into left & right views
      auto [left, right] = decompose(std::forward<A>(a));

      // get a nice name for this CPO
      auto compose = *this;
      
      // recursively compose A's right part with b and compose that result with A's left part
      return compose(left, compose(right, std::forward<B>(b)));
    }
    else if constexpr(viewable_tensor_like<A>)
    {
      return detail::make_composed_view(all(std::forward<A>(a)), all(std::forward<B>(b)));
    }
    else
    {
      // when A is not tensor_like (it could be a pointer or invocable), we don't call all(a)
      return detail::make_composed_view(std::forward<A>(a), all(std::forward<B>(b)));
    }
  }
};

} // end detail


namespace
{

constexpr detail::dispatch_compose compose;

} // end anonymous namespace


namespace detail
{

// compose and composed_tensor have a cyclic dependency (via composed_view) and can't use each other directly
// define detail::invoke_compose as soon as compose's definition is available
template<class... Args>
constexpr view auto invoke_compose(Args&&... args)
{
  return compose(std::forward<Args>(args)...);
}

} // end detail


} // end ubu

#include "../../detail/epilogue.hpp"

