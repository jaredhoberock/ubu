#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinate/concepts/congruent.hpp"
#include "../../coordinate/concepts/coordinate.hpp"
#include "../../coordinate/concepts/weakly_congruent.hpp"
#include "../../coordinate/coordinate_sum.hpp"
#include "../../coordinate/detail/as_integral.hpp"
#include "../../coordinate/detail/tuple_algorithm.hpp"
#include "apply_stride.hpp"
#include <concepts>


namespace ubu
{
namespace detail
{


template<std::integral R, scalar_coordinate D, scalar_coordinate C>
constexpr R apply_stride_r_impl(const D& stride, const C& coord)
{
  return static_cast<R>(as_integral(stride)) * static_cast<R>(as_integral(coord));
}


template<coordinate R, nonscalar_coordinate D, nonscalar_coordinate C>
  requires weakly_congruent<C,D>
constexpr auto apply_stride_r_impl(const D& stride, const C& coord);


template<nonscalar_coordinate R, nonscalar_coordinate D, scalar_coordinate C>
constexpr R apply_stride_r_impl(const D& stride, const C& coord)
{
  return detail::tuple_zip_with(stride, R{}, [&](const auto& s, auto r)
  {
    return apply_stride_r_impl<decltype(r)>(s, coord);
  });
}


template<coordinate R, nonscalar_coordinate D, nonscalar_coordinate C>
  requires weakly_congruent<C,D>
constexpr auto apply_stride_r_impl(const D& stride, const C& coord)
{
  auto star = [](const auto& s, const auto& c)
  {
    return apply_stride_r_impl<R>(s,c);
  };

  auto plus = [](const auto& c1, const auto& c2)
  {
    return coordinate_sum(c1,c2);
  };

  return detail::tuple_inner_product(stride, coord, star, plus);
}


} // end detail


// this variant of apply_stride allows control over the width of the result type, R
// R must be congruent to the result returned by the other variant, apply_stride()
template<coordinate R, coordinate D, weakly_congruent<D> C>
  requires (congruent<R, apply_stride_t<D,C>>
            and not std::is_reference_v<R>)
constexpr R apply_stride_r(const D& stride, const C& coord)
{
  return detail::apply_stride_r_impl<R>(stride, coord);
}


} // end ubu

#include "../../../detail/epilogue.hpp"

