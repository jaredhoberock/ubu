#pragma once

#include "../../detail/prologue.hpp"

#include "../concepts/composable.hpp"
#include "../concepts/decomposable.hpp"
#include "../concepts/tensor.hpp"
#include "../concepts/viewable_tensor.hpp"
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


template<class T>
concept viewable_tensor_or_not_tensor =
  viewable_tensor<T> or not tensor<T>
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


template<class A, class B>
concept has_compose_customization = has_compose_member_function<A,B> or has_compose_free_function<A,B>;


struct dispatch_compose
{
  template<class A, class B>
    requires has_compose_customization<A&&,B&&>
  constexpr view_of_composition<A&&,B&&> auto operator()(A&& a, B&& b) const
  {
    if constexpr (has_compose_member_function<A&&,B&&>)
    {
      return std::forward<A>(a).compose(std::forward<B>(b));
    }
    else
    {
      return compose(std::forward<A>(a), std::forward<B>(b));
    }
  }

  template<viewable_tensor_or_not_tensor A, tensor B>
    requires (not has_compose_customization<A&&,B&&>
              and composable<A,B>)
  constexpr view_of_composition<A&&,B&&> auto operator()(A&& a, B&& b) const
  {
    if constexpr (decomposable<A&&>)
    {
      // get a nice name for this CPO
      auto compose = *this;

      // decompose a into left & right views
      auto [left, right] = decompose(std::forward<A>(a));
      
      // recursively compose A's right part with b and compose that result with A's left part
      return compose(left, compose(right, std::forward<B>(b)));
    }
    else if constexpr(viewable_tensor<A>)
    {
      return composed_view(all(std::forward<A>(a)), all(std::forward<B>(b)));
    }
    else
    {
      // when A is not a tensor (it could be a pointer or invocable), we don't call all(a)
      return composed_view(std::forward<A>(a), all(std::forward<B>(b)));
    }
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_compose compose;

} // end anonymous namespace


} // end ubu

#include "../../detail/epilogue.hpp"

