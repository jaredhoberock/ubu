#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../utilities/integrals/integral_like.hpp"
#include "../../../utilities/tuples.hpp"
#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include "zeros.hpp"
#include <concepts>


namespace ubu
{
namespace detail
{


template<integral_like T>
constexpr auto successor_of_each_mode(const T& i)
{
  return i + 1;
}


template<tuples::tuple_like T>
constexpr auto successor_of_each_mode(const T& t)
{
  return tuples::zip_with(t, [](auto element)
  {
    return successor_of_each_mode(element);
  });
}


} // end detail


// note that the type of ones_v may differ from T because of fancy coordinates like constant
template<coordinate T>
constexpr congruent<T> auto ones_v = detail::successor_of_each_mode(zeros_v<T>);


} // end ubu

#include "../../../detail/epilogue.hpp"

