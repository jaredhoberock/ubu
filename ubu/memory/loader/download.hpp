#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/initial_happening.hpp"
#include "../../causality/wait.hpp"
#include "address.hpp"
#include "downloader.hpp"
#include "download_after.hpp"

namespace ubu
{

namespace detail
{


template<class D, class A1, class N, class A2>
concept has_download_member_function = requires(D d, A1 from, N n, A2 to)
{
  d.download(from, n, to);
};


template<class D, class A1, class N, class A2>
concept has_download_free_function = requires(D d, A1 from, N n, A2 to)
{
  download(d, from, n, to);
};


// this is the type of download
struct dispatch_download
{
  // this dispatch path calls the member function
  template<class D, class A1, class N, class A2>
    requires has_download_member_function<D&&,A1&&,N&&,A2&&>
  constexpr auto operator()(D&& d, A1&& from, N&& n, A2&& to) const
  {
    return std::forward<D>(d).download(std::forward<A1>(from), std::forward<N>(n), std::forward<A2>(to));
  }

  // this dispatch path calls the free function
  template<class D, class A1, class N, class A2>
    requires (not has_download_member_function<D&&,A1&&,N&&,A2&&>
              and has_download_free_function<D&&,A1&&,N&&,A2&&>)
  constexpr auto operator()(D&& d, A1&& from, N&& n, A2&& to) const
  {
    return download(std::forward<D>(d), std::forward<A1>(from), std::forward<N>(n), std::forward<A2>(to));
  }

  // default path calls download_after and wait
  template<downloader U, class A1, class N, class A2>
    requires (not has_download_member_function<U,A1,N,A2>
              and not has_download_free_function<U,A1,N,A2>)
  constexpr void operator()(U u, A1 from, N n, A2 to) const
  {
    wait(download_after(u, initial_happening(u), from, n, to));
  }
};


} // end detail


inline constexpr detail::dispatch_download download;


} // end ubu

#include "../../detail/epilogue.hpp"

