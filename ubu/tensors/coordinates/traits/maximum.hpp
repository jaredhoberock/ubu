#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../utilities/integrals/integral_like.hpp"
#include "../../../utilities/tuples.hpp"
#include "../concepts/coordinate.hpp"
#include <limits>

namespace ubu
{
namespace detail
{


template<coordinate C>
constexpr C maximum_of_each_mode(C coord)
{
  if constexpr (integral_like<C>)
  {
    return std::numeric_limits<C>::max();
  }
  else
  {
    return tuples::zip_with(coord, [](auto element)
    {
      return maximum_of_each_mode(element);
    });
  }
}

} // end detail


template<coordinate T>
constexpr T maximum_v = detail::maximum_of_each_mode(T{});

} // end ubu

#include "../../../detail/epilogue.hpp"

