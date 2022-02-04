#pragma once

#include "../detail/prologue.hpp"

#include "../event/event.hpp"
#include "../event/wait.hpp"
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


template<event... Events>
class event_tuple
{
  public:
    event_tuple(const event_tuple&) = delete;
    event_tuple(event_tuple&&) = default;

    event_tuple(Events&&... events)
      : events_{std::move(events)...}
    {}

    void wait()
    {
      wait_impl(std::integral_constant<std::size_t,0>{});
    }

  private:
    void wait_impl(std::integral_constant<std::size_t, sizeof...(Events)>) {}

    template<std::size_t i>
    void wait_impl(std::integral_constant<std::size_t,i>)
    {
      ASPERA_NAMESPACE::wait(std::get<i>(events_));
      wait_impl(std::integral_constant<std::size_t,i+1>{});
    }

    std::tuple<Events...> events_;
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

  // the default path for a single event simply moves the event
  template<executor Ex, event E>
    requires (!has_contingent_on_member_function<Ex&&,E&&> and
              !has_contingent_on_free_function<Ex&&,E&&> and
              std::is_move_constructible_v<std::remove_reference_t<E>>)
  constexpr std::remove_cvref_t<E> operator()(Ex&&, E&& event) const
  {
    return std::move(event);
  }

  // the default path for many events moves the events into an event_tuple
  template<executor Ex, event... Es>
    requires (!has_contingent_on_member_function<Ex&&,Es&&...> and
              !has_contingent_on_free_function<Ex&&,Es&&...> and
              std::conjunction_v<
                std::is_move_constructible<std::remove_reference_t<Es>>...
              >)
  constexpr event_tuple<std::remove_cvref_t<Es>...>
    operator()(Ex&&, Es&&... events) const
  {
    return {std::move(events)...};
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

#include "../detail/epilogue.hpp"

