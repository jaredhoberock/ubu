#pragma once

#include "../../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include <utility>

namespace ubu
{

namespace detail
{


template<class U, class B, class A1, class N, class A2>
concept has_upload_after_member_function = requires(U u, B before, A1 from, N n, A2 to)
{
  { u.upload_after(before, from, n, to) } -> happening;
};


template<class U, class B, class A1, class N, class A2>
concept has_upload_after_free_function = requires(U u, B before, A1 from, N n, A2 to)
{
  { upload_after(u, from, n, to) } -> happening;
};


// this is the type of upload_after
struct dispatch_upload_after
{
  template<class U, class B, class A1, class N, class A2>
    requires has_upload_after_member_function<U&&,B&&,A1&&,N&&,A2&&>
  constexpr auto operator()(U&& u, B&& before, A1&& from, N&& n, A2&& to) const
  {
    return std::forward<U>(u).upload_after(std::forward<B>(before), std::forward<A1>(from), std::forward<N>(n), std::forward<A2>(to));
  }


  template<class U, class B, class A1, class N, class A2>
    requires (not has_upload_after_member_function<U&&,B&&,A1&&,N&&,A2&&>
              and has_upload_after_free_function<U&&,B&&,A1&&,N&&,A2&&>)
  constexpr auto operator()(U&& u, B&& before, A1&& from, N&& n, A2&& to) const
  {
    return upload_after(std::forward<U>(u), std::forward<B>(before), std::forward<A1>(from), std::forward<N>(n), std::forward<A2>(to));
  }
};


} // end detail


inline constexpr detail::dispatch_upload_after upload_after;

template<class U, class B, class A1, class N, class A2>
using upload_after_result_t = decltype(upload_after(std::declval<U>(), std::declval<B>(), std::declval<A1>(), std::declval<N>(), std::declval<A2>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

