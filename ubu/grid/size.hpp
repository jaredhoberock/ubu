#pragma once

#include "../detail/prologue.hpp"
#include "shape/shape.hpp"
#include <ranges>

namespace ubu
{
namespace detail
{

template<class T>
concept has_shape = requires(T arg)
{
  ubu::shape(arg);
};

struct dispatch_size
{
  template<class T>
    requires has_ranges_size<T&&>
  constexpr std::integral auto operator()(T&& arg) const
  {
    return std::ranges::size(std::forward<T>(arg));
  }

  template<class T>
    requires (not has_ranges_size<T&&>
              and has_shape<T&&>)
  constexpr std::integral auto operator()(T&& arg) const
  {
    // XXX this is only correct when we can detect that
    //     arg "has no holes" i.e., if it has a mask, then
    //     all positions in its mask must be true
    return shape_size(shape(std::forward<T>(arg)));
  }
}; // end dispatch_size

} // end detail

constexpr detail::dispatch_size size;

} // end ubu

#include "../detail/epilogue.hpp"

