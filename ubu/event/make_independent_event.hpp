#pragma once

#include "../detail/prologue.hpp"

#include "event.hpp"
#include <concepts>
#include <type_traits>


namespace ubu
{

namespace detail
{


template<class T>
concept has_make_independent_event_member_function = requires(T arg)
{
  {arg.make_independent_event()} -> event;
};

template<class T>
concept has_make_independent_event_free_function = requires(T arg)
{
  {make_independent_event(arg)} -> event;
};

template<class T>
concept has_event_type_member_type = requires
{
  typename std::decay_t<T>::event_type;
  requires event<typename std::decay_t<T>::event_type>;
};

template<class T>
concept has_make_independent_event_static_member_function = requires
{
  requires event<T>;

  // the result of make_independent_event() be the same event type as T
  {T::make_independent_event()} -> std::same_as<T>;
};


struct dispatch_make_independent_event
{
  template<class T>
    requires has_make_independent_event_member_function<T>
  constexpr auto operator()(T&& arg) const
  {
    return std::forward<T>(arg).make_independent_event();
  }

  template<class T>
    requires (!has_make_independent_event_member_function<T> and
               has_make_independent_event_free_function<T>)
  constexpr auto operator()(T&& arg) const
  {
    return make_independent_event(std::forward<T>(arg));
  }

  template<class T>
    requires (!has_make_independent_event_free_function<T> and
              !has_make_independent_event_free_function<T> and
              has_event_type_member_type<T> and
              has_make_independent_event_static_member_function<typename std::decay_t<T>::event_type>)
  constexpr auto operator()(T&&) const
  {
    using event_type = typename std::decay_t<T>::event_type;
    return event_type::make_independent_event();
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_make_independent_event make_independent_event;

} // end anonymous namespace


template<class T>
using make_independent_event_result_t = decltype(ubu::make_independent_event(std::declval<T>()));


} // end ubu


#include "../detail/epilogue.hpp"

