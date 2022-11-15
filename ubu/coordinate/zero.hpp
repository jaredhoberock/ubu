#pragma once

#include "../detail/prologue.hpp"

#include "../detail/for_each_tuple_element.hpp"
#include "coordinate.hpp"
#include "element.hpp"


namespace ubu
{
namespace detail
{


template<scalar_coordinate T>
constexpr T zero()
{
  return 0;
}


template<nonscalar_coordinate T>
constexpr T zero()
{
  T result{};

  for_each_tuple_element([](auto& element)
  {
    element = zero<std::remove_cvref_t<decltype(element)>>();
  }, result);

  return result;
}


} // end detail


template<coordinate T>
constexpr T zero = detail::zero<T>();


} // end ubu

#include "../detail/epilogue.hpp"

