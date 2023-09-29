#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate.hpp"
#include <ranges>

namespace ubu
{
namespace detail
{


template<class T>
concept has_shape_member_function = requires(T arg)
{
  { arg.shape() } -> coordinate;
};

template<class T>
concept has_shape_free_function = requires(T arg)
{
  { shape(arg) } -> coordinate;
};

template<class T>
concept has_ranges_size = requires(T arg)
{
  { std::ranges::size(arg) } -> coordinate;
};


struct dispatch_shape
{
  template<class T>
    requires has_shape_member_function<T&&>
  constexpr coordinate auto operator()(T&& arg) const
  {
    return std::forward<T>(arg).shape();
  }

  template<class T>
    requires (not has_shape_member_function<T&&>
              and has_shape_free_function<T&&>)
  constexpr coordinate auto operator()(T&& arg) const
  {
    return shape(std::forward<T>(arg));
  }

  template<class T>
    requires (not has_shape_member_function<T&&>
              and not has_shape_free_function<T&&>
              and has_ranges_size<T&&>)
  constexpr coordinate auto operator()(T&& arg) const
  {
    return std::ranges::size(std::forward<T>(arg));
  }
}; // end dispatch_shape


} // end detail


namespace
{

constexpr detail::dispatch_shape shape;

} // end anonymous namespace



template<class T>
using shape_t = decltype(shape(std::declval<T>()));


} // end ubu

#include "../../detail/epilogue.hpp"
