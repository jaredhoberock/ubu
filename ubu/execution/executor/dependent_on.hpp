#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "../../event/make_dependent_event.hpp"
#include "../../event/wait.hpp"
#include "executor.hpp"
#include <type_traits>
#include <tuple>
#include <utility>


UBU_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class Ex, class... Events>
concept has_dependent_on_member_function = requires(Ex executor, Events... events)
{
  executor.dependent_on(events...);
};

template<class Ex, class... Events>
concept has_dependent_on_free_function = requires(Ex executor, Events... events)
{
  dependent_on(executor, events...);
};


// this is the type of dependent_on
struct dispatch_dependent_on
{
  // this dispatch path calls the member function
  template<class Ex, class... Events>
    requires has_dependent_on_member_function<Ex&&,Events&&...>
  constexpr auto operator()(Ex&& executor, Events&&... events) const
  {
    return std::forward<Ex>(executor).dependent_on(std::forward<Events>(events)...);
  }

  // this dispatch path calls the free function
  template<class Ex, class... Events>
    requires (!has_dependent_on_member_function<Ex&&,Events&&...> and
              has_dependent_on_free_function<Ex&&,Events&&...>)
  constexpr auto operator()(Ex&& executor, Events&&... events) const
  {
    return dependent_on(std::forward<Ex>(executor), std::forward<Events>(events)...);
  }

  // the default path drops the executor and calls make_dependent_event
  template<executor Ex, event... Events>
    requires (!has_dependent_on_member_function<Ex&&,Events&&...> and
              !has_dependent_on_free_function<Ex&&,Events&&...>)
  constexpr auto operator()(Ex&&, Events&&... events) const
  {
    return make_dependent_event(std::forward<Events>(events)...);
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_dependent_on dependent_on;

} // end anonymous namespace


template<class Ex, class... Events>
using dependent_on_result_t = decltype(UBU_NAMESPACE::dependent_on(std::declval<Ex>, std::declval<Events>...));


UBU_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

