#pragma once

#include "../../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include <utility>

namespace ubu
{
namespace detail
{


template<class D, class B, class A1, class N, class A2>
concept has_download_after_member_function = requires(D d, B before, A1 from, N n, A2 to)
{
  { d.download_after(before, from, n, to) } -> happening;
};


template<class D, class B, class A1, class N, class A2>
concept has_download_after_free_function = requires(D d, B before, A1 from, N n, A2 to)
{
  { download_after(d, before, from, n, to) } -> happening;
};


// this is the type of download_after
struct dispatch_download_after
{
  template<class D, class B, class A1, class N, class A2>
    requires has_download_after_member_function<D&&,B&&,A1&&,N&&,A2&&>
  constexpr auto operator()(D&& d, B&& before, A1&& from, N&& n, A2&& to) const
  {
    return std::forward<D>(d).download_after(std::forward<B>(before), std::forward<A1>(from), std::forward<N>(n), std::forward<A2>(to));
  }


  template<class D, class B, class A1, class N, class A2>
    requires (not has_download_after_member_function<D&&,B&&,A1&&,N&&,A2&&>
              and has_download_after_free_function<D&&,B&&,A1&&,N&&,A2&&>)
  constexpr auto operator()(D&& d, B&& before, A1&& from, N&& n, A2&& to) const
  {
    return download_after(std::forward<D>(d), std::forward<B>(before), std::forward<A1>(from), std::forward<N>(n), std::forward<A2>(to));
  }
};


} // end detail


inline constexpr detail::dispatch_download_after download_after;

template<class D, class B, class A1, class N, class A2>
using download_after_result_t = decltype(download_after(std::declval<D>(), std::declval<B>(), std::declval<A1>(), std::declval<N>(), std::declval<A2>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

