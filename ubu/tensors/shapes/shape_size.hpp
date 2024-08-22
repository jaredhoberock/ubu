#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/constant.hpp"
#include "../../utilities/integrals/integral_like.hpp"
#include "../../utilities/tuples.hpp"
#include "../coordinates/concepts/coordinate.hpp"
#include "../coordinates/detail/to_integral_like.hpp"
#include <concepts>


namespace ubu
{

template<coordinate S>
constexpr integral_like auto shape_size(const S& shape)
{
  if constexpr (unary_coordinate<S>)
  {
    return detail::to_integral_like(shape);
  }
  else
  {
    return tuples::fold_left(shape, 1_c, [](const auto& partial_product, const auto& s)
    {
      return partial_product * shape_size(s);
    });
  }
}

} // end ubu


#include "../../detail/epilogue.hpp"

