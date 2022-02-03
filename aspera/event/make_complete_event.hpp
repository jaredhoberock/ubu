#pragma once

#include "../detail/prologue.hpp"

#include "event.hpp"
#include <concepts>
#include <type_traits>

ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
concept has_make_complete_event_member_function = requires(T arg)
{
  {arg.make_complete_event()} -> event;
};

template<class T>
concept has_make_complete_event_free_function = requires(T arg)
{
  {make_complete_event(arg)} -> event;
};

template<class T>
concept has_event_type_member_type = requires
{
  typename std::decay_t<T>::event_type;
  requires event<typename std::decay_t<T>::event_type>;
};

template<class T>
concept has_make_complete_event_static_member_function = requires
{
  requires event<T>;

  // the result of make_complete_event() be the same event type as T
  {T::make_complete_event()} -> std::same_as<T>;
};


struct dispatch_make_complete_event
{
  template<class T>
    requires has_make_complete_event_member_function<T>
  constexpr auto operator()(T&& arg) const
  {
    return std::forward<T>(arg).make_complete_event();
  }

  template<class T>
    requires (!has_make_complete_event_member_function<T> and
               has_make_complete_event_free_function<T>)
  constexpr auto operator()(T&& arg) const
  {
    return make_complete_event(std::forward<T>(arg));
  }

  template<class T>
    requires (!has_make_complete_event_free_function<T> and
              !has_make_complete_event_free_function<T> and
              has_event_type_member_type<T> and
              has_make_complete_event_static_member_function<typename std::decay_t<T>::event_type>)
  constexpr auto operator()(T&&) const
  {
    using event_type = typename std::decay_t<T>::event_type;
    return event_type::make_complete_event();
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_make_complete_event make_complete_event;

} // end anonymous namespace


template<class T>
using make_complete_event_result_t = decltype(ASPERA_NAMESPACE::make_complete_event(std::declval<T>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

