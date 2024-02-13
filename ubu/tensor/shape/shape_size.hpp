#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate/concepts/coordinate.hpp"
#include "../coordinate/detail/as_integral.hpp"
#include "../coordinate/detail/tuple_algorithm.hpp"
#include <concepts>


namespace ubu
{


// scalar case
template<scalar_coordinate C>
constexpr std::integral auto shape_size(const C& shape)
{
  return detail::as_integral(shape);
}


// nonscalar case
template<nonscalar_coordinate C>
constexpr std::integral auto shape_size(const C& shape)
{
  return detail::tuple_fold(1, shape, [](const auto& partial_product, const auto& s)
  {
    return partial_product * shape_size(s);
  });
}


} // end ubu


#include "../../detail/epilogue.hpp"

