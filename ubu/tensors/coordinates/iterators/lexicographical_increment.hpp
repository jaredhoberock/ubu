#pragma once

#include "../../../detail/prologue.hpp"

#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include "../detail/to_integral_like.hpp"
#include "../traits/zeros.hpp"


namespace ubu
{


template<coordinate C, congruent<C> O, congruent<C> E>
  requires (0 <= rank_v<C> and rank_v<C> <= 1)
constexpr bool lexicographical_increment(C& coord, const O&, const E& end)
{
  if constexpr (nullary_coordinate<C>)
  {
    return true;
  }
  else
  {
    return ++detail::to_integral_like(coord) == end;
  }
}


template<multiary_coordinate C, congruent<C> O, congruent<C> E>
constexpr bool lexicographical_increment(C& coord, const O& origin, const E& end);


namespace detail
{


template<std::size_t I, multiary_coordinate C, congruent<C> O, congruent<C> E>
constexpr bool lexicographical_increment_impl(C& coord, const O& origin, const E& end)
{
  // increment the Ith element
  if(not lexicographical_increment(get<I>(coord), get<I>(origin), get<I>(end)))
  {
    // no roll over
    return false;
  }

  // continue to the left if there are more elements
  if constexpr (I > 0)
  {
    // note that we don't roll the final (0th) dimension over to the origin
    get<I>(coord) = get<I>(origin);

    // continue recursion towards the left
    return lexicographical_increment_impl<I-1>(coord, origin, end);
  }

  // coord is at the end
  return true;
}


} // end detail


template<multiary_coordinate C, congruent<C> O, congruent<C> E>
constexpr bool lexicographical_increment(C& coord, const O& origin, const E& end)
{
  return detail::lexicographical_increment_impl<rank_v<C> - 1>(coord, origin, end);
}


template<coordinate C, congruent<C> S>
constexpr bool lexicographical_increment(C& coord, const S& shape)
{
  return lexicographical_increment(coord, zeros_v<C>, shape);
}


} // end ubu


#include "../../../detail/epilogue.hpp"

