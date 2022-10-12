#pragma once

#include "../detail/prologue.hpp"

#include "../execution/policy.hpp"
#include "detail/default_for_each_n_after.hpp"

#include <iterator>
#include <utility>

namespace ubu
{

namespace detail
{


template<class P, class H, class I, class N, class F>
concept has_for_each_n_after_member_function = requires(P p, H h, I i, N n, F f)
{
  {p.for_each_n_after(h, i, n, f)} -> happening;
};

template<class P, class H, class I, class N, class F>
concept has_for_each_n_after_free_function = requires(P p, H h, I i, N n, F f)
{
  {for_each_n_after(p, h, i, n, f)} -> happening;
};


struct dispatch_for_each_n_after
{
  template<class P, happening H, class I, class N, class F>
    requires has_for_each_n_after_member_function<P&&,H&&,I&&,N&&,F&&>
  constexpr auto operator()(P&& p, H&& h, I&& i, N&& n, F&& f) const
  {
    return std::forward<P>(p).for_each_n_after(std::forward<H>(h), std::forward<I>(i), std::forward<N>(n), std::forward<F>(f));
  }

  template<class P, class H, class I, class N, class F>
    requires (!has_for_each_n_after_member_function<P&&,H&&,I&&,N&&,F&&> and
               has_for_each_n_after_free_function<P&&,H&&,I&&,N&&,F&&>)
  constexpr auto operator()(P&& p, H&& h, I&& i, N&& n, F&& f) const
  {
    return for_each_n_after(std::forward<P>(p), std::forward<H>(h), std::forward<I>(i), std::forward<N>(n), std::forward<F>(f));
  }

  // XXX H should be execution_policy_happening_t<P>
  template<execution_policy P, happening H, std::random_access_iterator I, std::indirectly_unary_invocable<I> F>
    requires (!has_for_each_n_after_member_function<P&&,H&&,I,std::iter_difference_t<I>,F> and
              !has_for_each_n_after_free_function<P&&,H&&,I,std::iter_difference_t<I>,F>)
  auto operator()(P&& policy, H&& before, I first, std::iter_difference_t<I> n, F f) const
  {
    return detail::default_for_each_n_after(std::forward<P>(policy), std::forward<H>(before), first, n, f);
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_for_each_n_after for_each_n_after;

}


} // end ubu

#include "../detail/epilogue.hpp"

