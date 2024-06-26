#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../miscellaneous/tuples.hpp"
#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include "../detail/to_integral_like.hpp"


namespace ubu
{


// is_below(lhs, rhs) returns true if all modes of lhs is < their corresponding mode of rhs

template<scalar_coordinate C1, scalar_coordinate C2>
constexpr bool is_below(const C1& lhs, const C2& rhs)
{
  return detail::to_integral_like(lhs) < detail::to_integral_like(rhs);
}

template<nonscalar_coordinate C1, nonscalar_coordinate C2>
  requires congruent<C1,C2>
constexpr bool is_below(const C1& lhs, const C2& rhs)
{
  auto tuple_of_results = tuples::zip_with(lhs, rhs, [](auto l, auto r)
  {
    return is_below(l, r);
  });

  return tuples::all_of(tuple_of_results, [](bool result)
  {
    return result;
  });
}


// is_below_or_equal(lhs, rhs) returns true if all modes of lhs is <= their corresponding mode of rhs

template<scalar_coordinate C1, scalar_coordinate C2>
constexpr bool is_below_or_equal(const C1& lhs, const C2& rhs)
{
  return detail::to_integral_like(lhs) <= detail::to_integral_like(rhs);
}

template<nonscalar_coordinate C1, nonscalar_coordinate C2>
  requires congruent<C1,C2>
constexpr bool is_below_or_equal(const C1& lhs, const C2& rhs)
{
  auto tuple_of_results = tuples::zip_with(lhs, rhs, [](auto l, auto r)
  {
    return is_below_or_equal(l, r);
  });

  return tuples::all_of(tuple_of_results, [](bool result)
  {
    return result;
  });
}


} // end ubu

#include "../../../detail/epilogue.hpp"

