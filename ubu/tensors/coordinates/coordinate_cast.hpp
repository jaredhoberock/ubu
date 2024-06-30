#pragma once

#include "../../detail/prologue.hpp"

#include "../../miscellaneous/tuples.hpp"
#include "concepts/congruent.hpp"
#include "concepts/coordinate.hpp"
#include "detail/to_integral_like.hpp"
#include <type_traits>


namespace ubu
{


template<scalar_coordinate T, scalar_coordinate C>
  requires (not std::is_reference_v<T>)
constexpr T coordinate_cast(const C& coord)
{
  return static_cast<T>(detail::to_integral_like(coord));
}

template<nonscalar_coordinate T, congruent<T> C>
  requires (not std::is_reference_v<T>)
constexpr T coordinate_cast(const C& coord)
{
  T zero{};

  return tuples::zip_with(zero, coord, [](auto z, const auto& c)
  {
    return coordinate_cast<decltype(z)>(c);
  });
}


} // end ubu

#include "../../detail/epilogue.hpp"

