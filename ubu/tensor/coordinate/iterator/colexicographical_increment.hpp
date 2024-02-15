#pragma once

#include "../../../detail/prologue.hpp"

#include "../compare/is_below.hpp"
#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include "../detail/as_integral_like.hpp"
#include "../zeros.hpp"


namespace ubu
{


template<scalar_coordinate C, congruent<C> O, congruent<C> E>
constexpr void colexicographical_increment(C& coord, const O&, const E&)
{
  ++detail::as_integral_like(coord);
}


template<nonscalar_coordinate C, congruent<C> O, congruent<C> E>
constexpr void colexicographical_increment(C& coord, const O& origin, const E& end);


namespace detail
{


template<std::size_t I, nonscalar_coordinate C, congruent<C> O, congruent<C> E>
constexpr void colexicographical_increment_impl(C& coord, const O& origin, const E& end)
{
  // recurse into the Ith element
  colexicographical_increment(get<I>(coord), get<I>(origin), get<I>(end));

  // check the Ith element against the Ith bounds
  if(is_below(get<I>(coord), get<I>(end)))
  {
    return;
  }

  // roll over the Ith element to the origin
  if constexpr (I < rank_v<C> - 1)
  {
    // note that we don't roll the final (rank-1) dimension over to the origin
    get<I>(coord) = get<I>(origin);

    // continue recursion towards the right
    colexicographical_increment_impl<I+1>(coord, origin, end);
  }
}


} // end detail


template<nonscalar_coordinate C, congruent<C> O, congruent<C> E>
constexpr void colexicographical_increment(C& coord, const O& origin, const E& end)
{
  return detail::colexicographical_increment_impl<0>(coord, origin, end);
}


template<coordinate C, congruent<C> S>
constexpr void colexicographical_increment(C& coord, const S& shape)
{
  return colexicographical_increment(coord, zeros<C>, shape);
}


} // end ubu


#include "../../../detail/epilogue.hpp"

