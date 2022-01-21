#pragma once

#include "../detail/prologue.hpp"

#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class Ex, class Ev, class S, class F>
concept has_bulk_execute_member_function = requires(Ex executor, Ev event, S shape, F function) { executor.bulk_execute(event, shape, function); };

template<class Ex, class Ev, class S, class F>
concept has_bulk_execute_free_function = requires(Ex executor, Ev event, S shape, F function) { bulk_execute(executor, event, shape, function); };


// this is the type of bulk_execute
struct dispatch_bulk_execute
{
  // this dispatch path calls the member function
  template<class Ex, class Ev, class S, class F>
    requires has_bulk_execute_member_function<Ex&&,Ev&&,S&&,F&&>
  constexpr auto operator()(Ex&& executor, Ev&& event, S&& shape, F&& function) const
  {
    return std::forward<Ex>(executor).bulk_execute(std::forward<Ev>(event), std::forward<S>(shape), std::forward<F>(function));
  }

  /// this dispatch path calls the free function
  template<class Ex, class Ev, class S, class F>
    requires (!has_bulk_execute_member_function<Ex&&,Ev&&,S&&,F&&> and has_bulk_execute_free_function<Ex&&,Ev&&,S&&,F&&>)
  constexpr auto operator()(Ex&& executor, Ev&& event, S&& shape, F&& function) const
  {
    return bulk_execute(std::forward<Ex>(executor), std::forward<Ev>(event), std::forward<S>(shape), std::forward<F>(function));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_bulk_execute bulk_execute;

} // end anonymous namespace


template<class Ex, class Ev, class S, class F>
using bulk_execute_result_t = decltype(ASPERA_NAMESPACE::bulk_execute(std::declval<Ex>(), std::declval<Ev>(), std::declval<S>(), std::declval<F>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

