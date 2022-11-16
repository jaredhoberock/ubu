#pragma once

#include "../detail/prologue.hpp"

#include "coordinate.hpp"
#include "element.hpp"
#include "zero.hpp"


namespace ubu
{


template<scalar_coordinate C>
constexpr void lexicographic_decrement(C& coord, const C&, const C&)
{
  --coord;
}


template<nonscalar_coordinate C>
constexpr void lexicographic_decrement(C& coord, const C& origin, const C& end);


namespace detail
{


template<std::size_t I, nonscalar_coordinate C>
constexpr void lexicographic_decrement_impl(C& coord, const C& origin, const C& end)
{
  // recurse into the Ith element
  lexicographic_decrement(element<I>(coord), element<I>(origin), element<I>(end));

  // check the Ith element against the Ith bounds
  if(element<I>(origin) <= element<I>(coord))
  {
    return;
  }

  // set the Ith element to the end
  element<I>(coord) = element<I>(end);

  // decrement the element one more time to offset us one from the end
  lexicographic_decrement(element<I>(coord), element<I>(origin), element<I>(end));

  if constexpr (I > 0)
  {
    // continue recursion towards the left
    lexicographic_decrement_impl<I-1>(coord, origin, end);
  }
}


} // end detail


template<nonscalar_coordinate C>
constexpr void lexicographic_decrement(C& coord, const C& origin, const C& end)
{
  return detail::lexicographic_decrement_impl<rank_v<C> - 1>(coord, origin, end);
}


template<coordinate C>
constexpr void lexicographic_decrement(C& coord, const C& shape)
{
  return lexicographic_decrement(coord, zero<C>, shape);
}


} // end ubu

#include "../detail/epilogue.hpp"

