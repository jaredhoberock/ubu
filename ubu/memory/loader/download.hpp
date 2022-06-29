#pragma once

#include "../../detail/prologue.hpp"

#include <concepts>

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
  template<class Downloader, class A1, class N, class A2>
    requires has_download_member_function<Downloader&&,A1&&,N&&,A2&&>
  constexpr auto operator()(Downloader&& d, A1&& from, N&& n, A2&& to) const
  {
    return std::forward<Downloader>(d).download(std::forward<A1>(from), std::forward<N>(n), std::forward<A2>(to));
  }


  template<class Downloader, class A1, class N, class A2>
    requires (!has_download_member_function<Downloader&&,A1&&,N&&,A2&&>
              and has_download_free_function<Downloader&&,A1&&,N&&,A2&&>)
  constexpr auto operator()(Downloader&& d, A1&& from, N&& n, A2&& to) const
  {
    return upload(std::forward<Downloader>(d), std::forward<A1>(from), std::forward<N>(n), std::forward<A2>(to));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_download download;

} // end anonymous namespace


} // end ubu

#include "../../detail/epilogue.hpp"

