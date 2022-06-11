#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include <concepts>

namespace ubu
{

namespace detail
{


template<class C, class A1, class N, class A2>
concept has_copy_n_member_function = requires(C c, A1 from, N n, A2 to)
{
  c.copy_n(from, n, to);
};


template<class C, class A1, class N, class A2>
concept has_copy_n_free_function = requires(C c, A1 from, N n, A2 to)
{
  copy_n(c, from, n, to);
};


// this is the type of copy_n
struct dispatch_copy_n
{
  template<class Copier, class A1, class N, class A2>
    requires has_copy_n_member_function<Copier&&,A1&&,N&&,A2&&>
  constexpr auto operator()(Copier&& c, A1&& from, N&& n, A2&& to) const
  {
    return std::forward<Copier>(c).copy_n(std::forward<A1>(from), std::forward<N>(n), std::forward<A2>(to));
  }


  template<class Copier, class A1, class N, class A2>
    requires (!has_copy_n_member_function<Copier&&,A1&&,N&&,A2&&>
              and has_copy_n_free_function<Copier&&,A1&&,N&&,A2&&>)
  constexpr auto operator()(Copier&& c, A1&& from, N&& n, A2&& to) const
  {
    return copy_n(std::forward<Copier>(c), std::forward<A1>(from), std::forward<N>(n), std::forward<A2>(to));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_copy_n copy_n;

} // end anonymous namespace


} // end ubu

#include "../../detail/epilogue.hpp"

