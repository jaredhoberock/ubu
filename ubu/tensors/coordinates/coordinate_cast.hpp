#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/tuples.hpp"
#include "concepts/congruent.hpp"
#include "concepts/coordinate.hpp"
#include "detail/to_integral_like.hpp"
#include <concepts>
#include <type_traits>


namespace ubu
{


// XXX "cast" is maybe a bad name because this function
//     may not return exactly a T (in some cases, this is impossible).
//     it promises only to return a result congruent to T,
//     but it does its best to produce a T
template<coordinate T, congruent<T> C>
  requires (not std::is_reference_v<T>)
constexpr congruent<T> auto coordinate_cast(const C& coord)
{
  if constexpr (scalar_coordinate<T>)
  {
    if constexpr (std::constructible_from<T,detail::to_integral_like_t<C>>)
    {
      return T(detail::to_integral_like(coord));
    }
    else
    {
      // this case may happen if T is a ubu::constant, for example
      return detail::to_integral_like(coord);
    }
  }
  else
  {
    T zero{};

    return tuples::zip_with(zero, coord, [](auto z, const auto& c)
    {
      return coordinate_cast<decltype(z)>(c);
    });
  }
}


} // end ubu

#include "../../detail/epilogue.hpp"

