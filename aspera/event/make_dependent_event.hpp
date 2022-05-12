#pragma once

#include "../detail/prologue.hpp"

#include "event.hpp"
#include <utility>
#include <type_traits>

ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class E, class... Es>
concept has_make_dependent_event_member_function = requires(E e, Es... es)
{
  {e.make_dependent_event(es...)} -> event;
};

template<class E, class... Es>
concept has_make_dependent_event_free_function = requires(E e, Es... es)
{
  {make_dependent_event(e,es...)} -> event;
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


struct dispatch_make_dependent_event
{
  template<event E, event... Es>
    requires has_make_dependent_event_member_function<E&&,Es&&...>
  constexpr auto operator()(E&& e, Es&&... es) const
  {
    return std::forward<E>(e).make_dependent_event(std::forward<Es>(es)...);
  }

  template<event E, event... Es>
    requires (!has_make_dependent_event_member_function<E&&,Es&&...> and
               has_make_dependent_event_free_function<E&&,Es&&...>)
  constexpr auto operator()(E&& e, Es&&... es) const
  {
    return make_dependent_event(std::forward<E>(e), std::forward<Es>(es)...);
  }


  // a single event 
  template<event E>
    requires (!has_make_dependent_event_member_function<E&&> and
              !has_make_dependent_event_free_function<E&&> and
              std::constructible_from<std::remove_cvref_t<E>, E&&>)
  constexpr std::remove_cvref_t<E> operator()(E&& e) const
  {
    return std::forward<E>(e);
  }


  // the default path for many events moves the events into an event_tuple
  template<event E, event... Es>
    requires (!has_make_dependent_event_member_function<E&&,Es&&...> and
              !has_make_dependent_event_free_function<E&&,Es&&...> and
              std::constructible_from<std::remove_cvref_t<E>,E&&> and
              std::conjunction_v<
                std::is_constructible<std::remove_cvref_t<Es>,Es&&>...
              >)
  constexpr event_tuple<std::remove_cvref_t<Es>...>
    operator()(Es&&... events) const
  {
    return {std::move(events)...};
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_make_dependent_event make_dependent_event;

} // end anonymous namespace


template<class E, class... Es>
using make_dependent_event_result_t = decltype(ASPERA_NAMESPACE::make_dependent_event(std::declval<E>(), std::declval<Es>()...));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

