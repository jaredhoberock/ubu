#pragma once

#include "../detail/prologue.hpp"

#include "happening.hpp"
#include <concepts>
#include <type_traits>


namespace ubu
{

namespace detail
{


template<class T>
concept has_first_cause_member_function = requires(T arg)
{
  {arg.first_cause()} -> happening;
};

template<class T>
concept has_first_cause_free_function = requires(T arg)
{
  {first_cause(arg)} -> happening;
};

template<class T>
concept has_happening_type_member_type = requires
{
  typename std::decay_t<T>::happening_type;
  requires happening<typename std::decay_t<T>::happening_type>;
};

template<class T>
concept has_first_cause_static_member_function = requires
{
  requires happening<T>;

  // the result of first_cause() be the same happening type as T
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
    requires (!has_first_cause_member_function<T> and
              !has_first_cause_free_function<T> and
              has_happening_type_member_type<T> and
              has_first_cause_static_member_function<typename std::decay_t<T>::happening_type>)
  constexpr auto operator()(T&&) const
  {
    using happening_type = typename std::decay_t<T>::happening_type;
    return happening_type::first_cause();
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

