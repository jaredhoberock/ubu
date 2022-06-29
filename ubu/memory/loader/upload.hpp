#pragma once

#include "../../detail/prologue.hpp"

#include <concepts>

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
  template<class Uploader, class A1, class N, class A2>
    requires has_upload_member_function<Uploader&&,A1&&,N&&,A2&&>
  constexpr auto operator()(Uploader&& u, A1&& from, N&& n, A2&& to) const
  {
    return std::forward<Uploader>(u).upload(std::forward<A1>(from), std::forward<N>(n), std::forward<A2>(to));
  }


  template<class Uploader, class A1, class N, class A2>
    requires (!has_upload_member_function<Uploader&&,A1&&,N&&,A2&&>
              and has_upload_free_function<Uploader&&,A1&&,N&&,A2&&>)
  constexpr auto operator()(Uploader&& u, A1&& from, N&& n, A2&& to) const
  {
    return upload(std::forward<Uploader>(u), std::forward<A1>(from), std::forward<N>(n), std::forward<A2>(to));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_upload upload;

} // end anonymous namespace


} // end ubu

#include "../../detail/epilogue.hpp"

