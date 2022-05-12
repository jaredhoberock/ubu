#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "../../event/make_contingent_event.hpp"
#include "../../event/wait.hpp"
#include "executor.hpp"
#include <type_traits>
#include <tuple>
#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class Ex, class... Events>
concept has_contingent_on_member_function = requires(Ex executor, Events... events)
{
  executor.contingent_on(events...);
};

template<class Ex, class... Events>
concept has_contingent_on_free_function = requires(Ex executor, Events... events)
{
  contingent_on(executor, events...);
};


// this is the type of contingent_on
struct dispatch_contingent_on
{
  // this dispatch path calls the member function
  template<class Ex, class... Events>
    requires has_contingent_on_member_function<Ex&&,Events&&...>
  constexpr auto operator()(Ex&& executor, Events&&... events) const
  {
    return std::forward<Ex>(executor).contingent_on(std::forward<Events>(events)...);
  }

  // this dispatch path calls the free function
  template<class Ex, class... Events>
    requires (!has_contingent_on_member_function<Ex&&,Events&&...> and
              has_contingent_on_free_function<Ex&&,Events&&...>)
  constexpr auto operator()(Ex&& executor, Events&&... events) const
  {
    return contingent_on(std::forward<Ex>(executor), std::forward<Events>(events)...);
  }

  // the default path drops the executor and calls make_contingent_event
  template<executor Ex, event... Events>
    requires (!has_contingent_on_member_function<Ex&&,Events&&...> and
              !has_contingent_on_free_function<Ex&&,Events&&...>)
  constexpr auto operator()(Ex&&, Events&&... events) const
  {
    return make_contingent_event(std::forward<Events>(events)...);
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_contingent_on contingent_on;

} // end anonymous namespace


template<class Ex, class... Events>
using contingent_on_result_t = decltype(ASPERA_NAMESPACE::contingent_on(std::declval<Ex>, std::declval<Events>...));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

