#pragma once

#include "../../detail/prologue.hpp"

#include <future>
#include <ranges>
#include <type_traits>
#include <utility>


namespace ubu
{

namespace detail
{


template<class T>
concept has_initial_happening_member_function = requires(T arg)
{
  // XXX this should require that the result is an object
  arg.initial_happening();
};

template<class T>
concept has_initial_happening_free_function = requires(T arg)
{
  // XXX this should require that the result is an object
  initial_happening(arg);
};

template<class T>
concept has_happening_type_member_type = requires
{
  // XXX this should require that this type is an object
  typename std::decay_t<T>::happening_type;
};

template<class T>
concept has_initial_happening_static_member_function = requires
{
  // XXX this should require that the result is an object
  T::initial_happening();
};

template<class T>
concept has_happening_type_member_type_with_initial_happening_static_member_function =
  has_happening_type_member_type<T>
  and has_initial_happening_static_member_function<typename std::decay_t<T>::happening_type>
;


template<class T>
concept has_initial_happening_customization =
  has_initial_happening_member_function<T>
  or has_initial_happening_free_function<T>
  or has_happening_type_member_type_with_initial_happening_static_member_function<T>
;


struct dispatch_initial_happening
{
  template<class T>
    requires has_initial_happening_member_function<T>
  constexpr auto operator()(T&& arg) const
  {
    return std::forward<T>(arg).initial_happening();
  }

  template<class T>
    requires (not has_initial_happening_member_function<T>
              and has_initial_happening_free_function<T>)
  constexpr auto operator()(T&& arg) const
  {
    return initial_happening(std::forward<T>(arg));
  }

  template<class T>
    requires (    not has_initial_happening_member_function<T>
              and not has_initial_happening_free_function<T>
              and has_happening_type_member_type_with_initial_happening_static_member_function<T>)
  constexpr auto operator()(T&&) const
  {
    using happening_type = typename std::decay_t<T>::happening_type;
    return happening_type::initial_happening();
  }

  // customization for std::future<void>
  // XXX per the comment in after_all.hpp, rather than provide this customization for std::future<void>,
  // maybe we should just implement a bare-bones c++ event type using standard synchronization primitives
  // and allow std::future<void> to be created from that
  inline std::future<void> operator()(const std::future<void>&) const
  {
    std::promise<void> p;
    p.set_value();
    return p.get_future();
  }

  // customization for a range of happenings
  // XXX this is incorrect because initial_happening needs to return the same type as its argument
  template<std::ranges::range R>
    requires (not has_initial_happening_customization<R&&>
              and has_initial_happening_customization<std::ranges::range_value_t<R&&>>)
  constexpr auto operator()(R&& rng) const
  {
    return (*this)(*std::ranges::begin(rng));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_initial_happening initial_happening;

} // end anonymous namespace


template<class T>
using initial_happening_result_t = decltype(initial_happening(std::declval<T>()));


} // end ubu


#include "../../detail/epilogue.hpp"

