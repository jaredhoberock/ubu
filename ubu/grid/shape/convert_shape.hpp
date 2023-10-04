#pragma once

#include "../../detail/prologue.hpp"

#include "detail/approximate_factors.hpp"
#include "shape_size.hpp"
#include "../coordinate/coordinate.hpp"
#include "../coordinate/coordinate_cast.hpp"
#include "../coordinate/detail/tuple_algorithm.hpp"
#include "../coordinate/zeros.hpp"
#include <array>

namespace ubu
{

// convert_shape converts a shape from type S into type T
// postcondition: shape_size(result) >= shape_size(shape)
//
// if weakly_congruent<T,S>, shape_size(result) == shape_size(shape)

// "lateral" cast to congruent shape
// when the requested type T is congruent to S, just cast to T
template<coordinate T, congruent<T> S>
constexpr T convert_shape(const S& shape)
{
  return coordinate_cast<T>(shape);
}

// downcast to scalar
template<scalar_coordinate T, nonscalar_coordinate S>
constexpr T convert_shape(const S& shape)
{
  return coordinate_cast<T>(shape_size(shape));
}

// upcast from scalar
// when the shape is an integral, factor it and recurse
template<nonscalar_coordinate T, scalar_coordinate S>
constexpr T convert_shape(const S& shape)
{
  // we need to go from a rank 1 shape to a type with higher rank
  constexpr std::size_t N = rank_v<T>;

  // find an approximate factorization of shape
  std::array<element_t<0,S>, N> factors = detail::approximate_factors<N>(element<0>(shape));

  // recurse with each factor
  return detail::tuple_zip_with(zeros<T>, factors, [](auto z, const auto& f)
  {
    return convert_shape<decltype(z)>(f);
  });
}

// upcast from nonscalar case 1
// when weakly_congruent<S,T>, recurse across dimensions
template<nonscalar_coordinate T, nonscalar_coordinate S>
  requires (not congruent<T,S> and weakly_congruent<S,T>)
constexpr T convert_shape(const S& shape)
{
  return detail::tuple_zip_with(zeros<T>, shape, [](auto z, const auto& s)
  {
    return convert_shape<decltype(z)>(s);
  });
}

// upcast from nonscalar case 2
// when not weakly_congruent<S,T>, collapse the shape to an integer and recurse
template<nonscalar_coordinate T, nonscalar_coordinate S>
  requires (not congruent<T,S> and not weakly_congruent<S,T>)
constexpr T convert_shape(const S& shape)
{
  return convert_shape<T>(shape_size(shape));
}


} // end ubu

#include "../../detail/epilogue.hpp"

