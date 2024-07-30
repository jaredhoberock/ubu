#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/tuples.hpp"
#include "concepts/congruent.hpp"
#include "concepts/coordinate.hpp"
#include "concepts/semicoordinate.hpp"
#include "detail/to_integral_like.hpp"
#include "traits/rank.hpp"
#include <concepts>
#include <type_traits>


namespace ubu
{


// XXX "cast" is maybe a bad name because this function
//     may not return exactly a T (in some cases, this is impossible).
//     it promises only to return a result congruent to T,
//     but it does its best to produce a T
template<semicoordinate T, congruent<T> C>
  requires (not std::is_reference_v<T>)
constexpr congruent<T> auto coordinate_cast(const C& coord)
{
  if constexpr (ubu::rank_v<T> == 1)
  {
    // if T is directly constructible from C, just do that
    if constexpr (std::constructible_from<T,const C&>)
    {
      return T(coord);
    }

    // else, check if C is a full coordinate (not just a semicoordinate)
    else if constexpr (scalar_coordinate<C>)
    {
      if constexpr (std::constructible_from<T,detail::to_integral_like_t<C>>)
      {
        // convert coord to an integral_like before constructing the T
        return T(detail::to_integral_like(coord));
      }
      else
      {
        // no conversion is possible; return coord's integral_like representation
        return detail::to_integral_like(coord);
      }
    }

    // no conversion is possible; just return coord
    else
    {
      return coord;
    }
  }
  else
  {
    // both T and C are tuple_like
    static_assert(tuples::tuple_like<T> and tuples::tuple_like<C>);

    T zero{};

    return tuples::zip_with(zero, coord, [](auto z, const auto& c)
    {
      return coordinate_cast<decltype(z)>(c);
    });
  }
}


} // end ubu

#include "../../detail/epilogue.hpp"

