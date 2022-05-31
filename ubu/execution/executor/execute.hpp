#pragma once

#include "../../detail/prologue.hpp"

#include <utility>


UBU_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class E, class F>
concept has_execute_member_function = requires(E e, F f) { e.execute(f); };

template<class E, class F>
concept has_execute_free_function = requires(E e, F f) { execute(e, f); };


// this is the type of execute
struct dispatch_execute
{
  // this dispatch path calls the member function
  template<class E, class F>
    requires has_execute_member_function<E&&,F&&>
  constexpr auto operator()(E&& e, F&& f) const
  {
    return std::forward<E>(e).execute(std::forward<F>(f));
  }

  // this dispatch path calls the free function
  template<class E, class F>
    requires (!has_execute_member_function<E&&,F&&> and has_execute_free_function<E&&,F&&>)
  constexpr auto operator()(E&& e, F&& f) const
  {
    return execute(std::forward<E>(e), std::forward<F>(f));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_execute execute;

} // end anonymous namespace


template<class E, class F>
using execute_result_t = decltype(UBU_NAMESPACE::execute(std::declval<E>(), std::declval<F>()));


UBU_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

