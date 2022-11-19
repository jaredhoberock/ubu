#pragma once

#include "../detail/prologue.hpp"

#include "coordinate.hpp"
#include "detail/colexicographic_coordinate.hpp"
#include "element.hpp"
#include "is_bounded_by.hpp"
#include "zeros.hpp"


namespace ubu
{


template<scalar_coordinate C>
constexpr void decrement_coordinate(C& coord, const C&, const C&)
{
  --coord;
}


template<nonscalar_coordinate C>
constexpr void decrement_coordinate(C& coord, const C& origin, const C& end);


namespace detail
{


template<std::size_t I, nonscalar_coordinate C>
constexpr void lexicographic_decrement(C& coord, const C& origin, const C& end)
{
  // recurse into the Ith element
  decrement_coordinate(element<I>(coord), element<I>(origin), element<I>(end));

  // check the Ith element against the Ith bounds
  if(is_bounded_by(element<I>(coord), element<I>(origin), element<I>(end)))
  {
    return;
  }

  // set the Ith element to the end
  element<I>(coord) = element<I>(end);

  // decrement the element one more time to offset us one from the end
  decrement_coordinate(element<I>(coord), element<I>(origin), element<I>(end));

  if constexpr (I > 0)
  {
    // continue recursion towards the left
    lexicographic_decrement<I-1>(coord, origin, end);
  }
}


template<std::size_t I, nonscalar_coordinate C>
constexpr void colexicographic_decrement(C& coord, const C& origin, const C& end)
{
  // recurse into the Ith element
  decrement_coordinate(element<I>(coord), element<I>(origin), element<I>(end));

  // check the Ith element against the Ith bounds
  if(is_bounded_by(element<I>(coord), element<I>(origin), element<I>(end)))
  {
    return;
  }

  // set the Ith element to the end
  element<I>(coord) = element<I>(end);

  // decrement the element one more time to offset us one from the end
  decrement_coordinate(element<I>(coord), element<I>(origin), element<I>(end));

  if constexpr (I > rank_v<C> - 1)
  {
    // continue recursion towards the right
    colexicographic_decrement<I+1>(coord, origin, end);
  }
}


} // end detail


template<nonscalar_coordinate C>
constexpr void decrement_coordinate(C& coord, const C& origin, const C& end)
{
  if constexpr(detail::colexicographic_coordinate<C>)
  {
    // when we can detect that C compares colexicographically, we decrement colexicographically
    return detail::colexicographic_decrement<0>(coord, origin, end);
  }
  else
  {
    // otherwise, we decrement lexicographically
    return detail::lexicographic_decrement<rank_v<C> - 1>(coord, origin, end);
  }
}


template<coordinate C>
constexpr void decrement_coordinate(C& coord, const C& shape)
{
  return decrement_coordinate(coord, zeros<C>, shape);
}


} // end ubu

#include "../detail/epilogue.hpp"

