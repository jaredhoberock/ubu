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
constexpr bool colexicographical_increment(C& coord, const O&, const E& end)
{
  if constexpr (rank_v<C> == 0)
  {
    return true;
  }
  else
  {
    return ++detail::to_integral_like(coord) == end;
  }
}


template<coordinate C, congruent<C> O, congruent<C> E>
  requires (rank_v<C> > 1)
constexpr bool colexicographical_increment(C& coord, const O& origin, const E& end);


namespace detail
{


template<std::size_t I, nonscalar_coordinate C, congruent<C> O, congruent<C> E>
constexpr bool colexicographical_increment_impl(C& coord, const O& origin, const E& end)
{
  // increment the Ith element
  if(not colexicographical_increment(get<I>(coord), get<I>(origin), get<I>(end)))
  {
    // no roll over
    return false;
  }

  // continue to the right if there are more elements
  if constexpr (I < rank_v<C> - 1)
  {
    // note that we don't roll the final (rank-1) dimension over to the origin
    get<I>(coord) = get<I>(origin);

    // recurse on element I+1
    return colexicographical_increment_impl<I+1>(coord, origin, end);
  }

  // coord is at the end
  return true;
}


} // end detail


template<coordinate C, congruent<C> O, congruent<C> E>
  requires (rank_v<C> > 1)
constexpr bool colexicographical_increment(C& coord, const O& origin, const E& end)
{
  return detail::colexicographical_increment_impl<0>(coord, origin, end);
}


template<coordinate C, congruent<C> S>
constexpr bool colexicographical_increment(C& coord, const S& shape)
{
  return colexicographical_increment(coord, zeros_v<C>, shape);
}


} // end ubu


#include "../../../detail/epilogue.hpp"

