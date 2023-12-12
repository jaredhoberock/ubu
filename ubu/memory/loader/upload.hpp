#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/initial_happening.hpp"
#include "../../causality/wait.hpp"
#include "address.hpp"
#include "uploader.hpp"
#include "upload_after.hpp"

namespace ubu
{

namespace detail
{


template<class U, class A1, class N, class A2>
concept has_upload_member_function = requires(U u, A1 from, N n, A2 to)
{
  u.upload(from, n, to);
};


template<class U, class A1, class N, class A2>
concept has_upload_free_function = requires(U u, A1 from, N n, A2 to)
{
  upload(u, from, n, to);
};


// this is the type of upload
struct dispatch_upload
{
  // this dispatch path calls the member function
  template<class U, class A1, class N, class A2>
    requires has_upload_member_function<U&&,A1&&,N&&,A2&&>
  constexpr auto operator()(U&& u, A1&& from, N&& n, A2&& to) const
  {
    return std::forward<U>(u).upload(std::forward<A1>(from), std::forward<N>(n), std::forward<A2>(to));
  }

  // this dispatch path calls the free function
  template<class U, class A1, class N, class A2>
    requires (not has_upload_member_function<U&&,A1&&,N&&,A2&&>
              and has_upload_free_function<U&&,A1&&,N&&,A2&&>)
  constexpr auto operator()(U&& u, A1&& from, N&& n, A2&& to) const
  {
    return upload(std::forward<U>(u), std::forward<A1>(from), std::forward<N>(n), std::forward<A2>(to));
  }

  // default path calls upload_after and wait
  template<uploader U, class A1, class N, class A2>
    requires (not has_upload_member_function<U,A1,N,A2>
              and not has_upload_free_function<U,A1,N,A2>)
  constexpr void operator()(U u, A1 from, N n, A2 to) const
  {
    wait(upload_after(u, initial_happening(u), from, n, to));
  }
};


} // end detail


inline constexpr detail::dispatch_upload upload;


} // end ubu

#include "../../detail/epilogue.hpp"

