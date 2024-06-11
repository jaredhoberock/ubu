#pragma once

#include "../../detail/prologue.hpp"

#include "actual_happening.hpp"
#include <utility>


namespace ubu
{

namespace detail
{


template<class H>
concept has_wait_member_function = requires(H h) { h.wait(); };

template<class H>
concept has_wait_free_function = requires(H h) { wait(h); };


// this is the type of wait
struct dispatch_wait
{
  // this dispatch path calls the member function
  template<class H>
    requires has_wait_member_function<H&&>
  constexpr auto operator()(H&& h) const
  {
    return std::forward<H>(h).wait();
  }

  // this dispatch path calls the free function
  template<class H>
    requires (!has_wait_member_function<H&&> and has_wait_free_function<H&&>)
  constexpr auto operator()(H&& h) const
  {
    return wait(std::forward<H>(h));
  }

  template<actual_happening H>
    requires (!has_wait_member_function<H&&> and !has_wait_free_function<H&&>)
  constexpr auto operator()(H&& h) const
  {
    // XXX optimize this
    while(not has_happened(h))
    {
      // busy wait
    }
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_wait wait;

} // end anonymous namespace


template<class E>
using wait_result_t = decltype(ubu::wait(std::declval<E>()));


} // end ubu


#include "../../detail/epilogue.hpp"

