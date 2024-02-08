#pragma once

#include "../detail/prologue.hpp"

#include "grid.hpp"
#include "layout/layout.hpp"
#include "view.hpp"
#include <concepts>
#include <utility>

namespace ubu
{
namespace detail
{

template<class R, class A, class B>
concept composition_of_grids =
  grid<R>
  and layout_for<B,A>
  and std::same_as<shape_t<R>, shape_t<B>>
;

template<class A, class B>
concept has_compose_member_function = requires(A a, B b)
{
  { a.compose(b) } -> composition_of_grids<A,B>;
};

template<class A, class B>
concept has_compose_free_function = requires(A a, B b)
{
  { compose(a,b) } -> composition_of_grids<A,B>;
};

struct dispatch_compose
{
  template<class A, class B>
    requires has_compose_member_function<A&&,B&&>
  constexpr grid auto operator()(A&& a, B&& b) const
  {
    return std::forward<A>(a).compose(std::forward<B>(b));
  }

  template<class A, class B>
    requires (not has_compose_member_function<A&&,B&&>
              and has_compose_free_function<A&&,B&&>)
  constexpr grid auto operator()(A&& a, B&& b) const
  {
    return compose(std::forward<A>(a), std::forward<B>(b));
  }

  template<class A, layout_for<A> B>
    requires (not has_compose_member_function<A&&,B&&>
              and not has_compose_free_function<A&&,B&&>)
  constexpr grid auto operator()(A&& a, B&& b) const
  {
    return view(std::forward<A>(a), std::forward<B>(b));
  }
};

} // end detail

namespace
{

constexpr detail::dispatch_compose compose;

} // end anonymous namespace

} // end ubu

#include "../detail/epilogue.hpp"

