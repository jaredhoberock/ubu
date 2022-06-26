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
concept has_first_cause_member_function = requires(T arg)
{
  {arg.first_cause()} -> event;
};

template<class T>
concept has_first_cause_free_function = requires(T arg)
{
  {first_cause(arg)} -> event;
};

template<class T>
concept has_event_type_member_type = requires
{
  typename std::decay_t<T>::event_type;
  requires event<typename std::decay_t<T>::event_type>;
};

template<class T>
concept has_first_cause_static_member_function = requires
{
  requires event<T>;

  // the result of first_cause() be the same event type as T
  {T::first_cause()} -> std::same_as<T>;
};


struct dispatch_first_cause
{
  template<class T>
    requires has_first_cause_member_function<T>
  constexpr auto operator()(T&& arg) const
  {
    return std::forward<T>(arg).first_cause();
  }

  template<class T>
    requires (!has_first_cause_member_function<T> and
               has_first_cause_free_function<T>)
  constexpr auto operator()(T&& arg) const
  {
    return first_cause(std::forward<T>(arg));
  }

  template<class T>
    requires (!has_first_cause_free_function<T> and
              !has_first_cause_free_function<T> and
              has_event_type_member_type<T> and
              has_first_cause_static_member_function<typename std::decay_t<T>::event_type>)
  constexpr auto operator()(T&&) const
  {
    using event_type = typename std::decay_t<T>::event_type;
    return event_type::first_cause();
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_first_cause first_cause;

} // end anonymous namespace


template<class T>
using first_cause_result_t = decltype(ubu::first_cause(std::declval<T>()));


} // end ubu


#include "../detail/epilogue.hpp"

