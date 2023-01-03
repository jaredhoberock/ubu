#pragma once

#include "../../detail/prologue.hpp"

#include "coordinate.hpp"
#include "element.hpp"
#include "detail/tuple_algorithm.hpp"
#include <concepts>


namespace ubu
{


// scalar case
template<scalar_coordinate C>
constexpr std::integral auto grid_size(const C& grid_shape)
{
  return element<0>(grid_shape);
}


// nonscalar case
template<nonscalar_coordinate C>
constexpr std::integral auto grid_size(const C& grid_shape)
{
  return detail::tuple_fold(1, grid_shape, [](const auto& partial_product, const auto& s)
  {
    return partial_product * grid_size(s);
  });
}


} // end ubu


#include "../../detail/epilogue.hpp"

