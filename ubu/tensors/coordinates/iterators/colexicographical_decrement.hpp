#pragma once

#include "../../../detail/prologue.hpp"

#include "../comparisons/is_below.hpp"
#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include "../detail/to_integral_like.hpp"
#include "../zeros.hpp"


namespace ubu
{


template<scalar_coordinate C, congruent<C> O, congruent<C> E>
constexpr void colexicographical_decrement(C& coord, const O&, const E&)
{
  --detail::to_integral_like(coord);
}


template<nonscalar_coordinate C, congruent<C> O, congruent<C> E>
constexpr void colexicographical_decrement(C& coord, const O& origin, const E& end);


namespace detail
{


template<std::size_t I, nonscalar_coordinate C, congruent<C> O, congruent<C> E>
constexpr void colexicographical_decrement_impl(C& coord, const O& origin, const E& end)
{
  // is the Ith element of coord at the origin?
  if(is_below_or_equal(get<I>(coord), get<I>(origin)))
  {
    // set the Ith element to the end
    get<I>(coord) = get<I>(end);

    // decrement the element one more time to offset us one from the end
    colexicographical_decrement(get<I>(coord), get<I>(origin), get<I>(end));

    if constexpr (I > rank_v<C> - 1)
    {
      // continue recursion towards the right
      colexicographical_decrement_impl<I+1>(coord, origin, end);
    }
  }
  else
  {
    colexicographical_decrement(get<I>(coord), get<I>(origin), get<I>(end));
  }
}


} // end detail


template<nonscalar_coordinate C, congruent<C> O, congruent<C> E>
constexpr void colexicographical_decrement(C& coord, const O& origin, const E& end)
{
  return detail::colexicographical_decrement_impl<0>(coord, origin, end);
}


template<coordinate C, congruent<C> S>
constexpr void colexicographical_decrement(C& coord, const S& shape)
{
  return colexicographical_decrement(coord, zeros<C>, shape);
}


} // end ubu

#include "../../../detail/epilogue.hpp"

