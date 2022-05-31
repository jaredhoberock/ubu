#pragma once

#include "../detail/prologue.hpp"

#include <utility>


UBU_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class E>
concept has_wait_member_function = requires(E e) { e.wait(); };

template<class E>
concept has_wait_free_function = requires(E e) { wait(e); };


// this is the type of wait
struct dispatch_wait
{
  // this dispatch path calls the member function
  template<class E>
    requires has_wait_member_function<E&&>
  constexpr auto operator()(E&& e) const
  {
    return std::forward<E>(e).wait();
  }

  // this dispatch path calls the free function
  template<class E>
    requires (!has_wait_member_function<E&&> and has_wait_free_function<E&&>)
  constexpr auto operator()(E&& e) const
  {
    return wait(std::forward<E>(e));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_wait wait;

} // end anonymous namespace


template<class E>
using wait_result_t = decltype(UBU_NAMESPACE::wait(std::declval<E>()));


UBU_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

