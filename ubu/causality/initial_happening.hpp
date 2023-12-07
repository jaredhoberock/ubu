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
concept has_initial_happening_member_function = requires(T arg)
{
  {arg.initial_happening()} -> happening;
};

template<class T>
concept has_initial_happening_free_function = requires(T arg)
{
  {initial_happening(arg)} -> happening;
};

template<class T>
concept has_happening_type_member_type = requires
{
  typename std::decay_t<T>::happening_type;
  requires happening<typename std::decay_t<T>::happening_type>;
};

template<class T>
concept has_initial_happening_static_member_function = requires
{
  requires happening<T>;

  // the result of initial_happening() be the same happening type as T
  {T::initial_happening()} -> std::same_as<T>;
};


struct dispatch_initial_happening
{
  template<class T>
    requires has_initial_happening_member_function<T>
  constexpr auto operator()(T&& arg) const
  {
    return std::forward<T>(arg).initial_happening();
  }

  template<class T>
    requires (!has_initial_happening_member_function<T> and
               has_initial_happening_free_function<T>)
  constexpr auto operator()(T&& arg) const
  {
    return initial_happening(std::forward<T>(arg));
  }

  template<class T>
    requires (!has_initial_happening_member_function<T> and
              !has_initial_happening_free_function<T> and
              has_happening_type_member_type<T> and
              has_initial_happening_static_member_function<typename std::decay_t<T>::happening_type>)
  constexpr auto operator()(T&&) const
  {
    using happening_type = typename std::decay_t<T>::happening_type;
    return happening_type::initial_happening();
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_initial_happening initial_happening;

} // end anonymous namespace


template<class T>
using initial_happening_result_t = decltype(ubu::initial_happening(std::declval<T>()));


} // end ubu


#include "../detail/epilogue.hpp"

