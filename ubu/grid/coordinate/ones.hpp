#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/for_each_tuple_element.hpp"
#include "concepts/coordinate.hpp"
#include <type_traits>


namespace ubu
{
namespace detail
{


template<scalar_coordinate T>
constexpr T ones_impl()
{
  return 1;
}


template<nonscalar_coordinate T>
constexpr T ones_impl()
{
  T result{};

  detail::for_each_tuple_element([](auto& element)
  {
    element = ones_impl<std::remove_cvref_t<decltype(element)>>();
  }, result);

  return result;
}


} // end detail


template<coordinate T>
constexpr std::remove_cvref_t<T> ones = detail::ones_impl<std::remove_cvref_t<T>>();


} // end ubu

#include "../../detail/epilogue.hpp"

