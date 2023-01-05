#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate/coordinate.hpp"
#include "../coordinate/coordinate_cast.hpp"
#include "../coordinate/detail/tuple_algorithm.hpp"
#include "../coordinate/grid_size.hpp"
#include "../coordinate/zeros.hpp"

namespace ubu
{


template<coordinate T, congruent<T> S>
constexpr T shape_cast(const S& shape)
{
  return coordinate_cast<T>(shape);
}

template<scalar_coordinate T, nonscalar_coordinate S>
constexpr T shape_cast(const S& shape)
{
  return static_cast<T>(grid_size(shape));
}

template<nonscalar_coordinate T, nonscalar_coordinate S>
  requires weakly_congruent<T,S>
constexpr T shape_cast(const S& shape)
{
  R z = zeros<T>;

  return detail::tuple_zip_with(z, shape, [](auto z, const auto& s)
  {
    return shape_cast<decltype(z)>(s);
  });
}


} // end ubu

#include "../../detail/epilogue.hpp"

