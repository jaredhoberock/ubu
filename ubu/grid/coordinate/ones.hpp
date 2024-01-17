#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/for_each_tuple_element.hpp"
#include "concepts/coordinate.hpp"
#include "zeros.hpp"
#include <concepts>


namespace ubu
{
namespace detail
{


template<std::integral T>
constexpr auto successor_of_each_mode(const T& i)
{
  return i + 1;
}


template<tuple_like T>
constexpr auto successor_of_each_mode(const T& t)
{
  return detail::tuple_zip_with(t, [](auto element)
  {
    return successor_of_each_mode(element);
  });
}


} // end detail


template<coordinate T>
constexpr auto ones = detail::successor_of_each_mode(zeros<T>);


} // end ubu

#include "../../detail/epilogue.hpp"

