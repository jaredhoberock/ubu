#pragma once

#include "../../detail/prologue.hpp"

#include "../../miscellaneous/tuples.hpp"
#include "../coordinates/concepts/congruent.hpp"
#include "../coordinates/concepts/coordinate.hpp"
#include "../shapes/shape.hpp"
#include "../shapes/shape_size.hpp"
#include "../views/compose.hpp"
#include <ranges>

namespace ubu
{


template<coordinate_of_rank<2> S>
class transposing_layout : public std::ranges::view_base
{
  public:
    constexpr transposing_layout(const S& transposed_shape)
      : transposed_shape_{transposed_shape}
    {}

    transposing_layout(const transposing_layout&) = default;

    template<congruent<S> C>
    constexpr S operator[](C coord) const
    {
      return S{get<1>(coord), get<0>(coord)};
    }

    constexpr S shape() const
    {
      return transposed_shape_;
    }

    constexpr std::size_t size() const
    {
      return shape_size(shape());
    }

  private:
    S transposed_shape_;
};


template<matrix_like M>
constexpr matrix_like auto transpose(M&& matrix)
{
  auto transposed_shape = tuples::reverse(shape(matrix));

  return compose(std::forward<M>(matrix), transposing_layout(transposed_shape));
}


} // end ubu

#include "../../detail/epilogue.hpp"

