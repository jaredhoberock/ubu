#pragma once

#include "prologue.hpp"

#include <functional>
#include <tuple>
#include <utility>


namespace ubu::detail
{


template<class... Args>
constexpr void ignore_arguments(Args&&...){}


template<class F, class T, std::size_t... I>
constexpr void for_each_tuple_element_impl(F&& f, T&& t, std::index_sequence<I...>)
{
  // the business with the trailing comma zero avoids problems with void-returning f
  ignore_arguments(
    (std::invoke(std::forward<F>(f), get<I>(std::forward<T>(t))), 0)...
  );
}


template<class F, class Tuple>
constexpr void for_each_tuple_element(F&& f, Tuple&& t)
{
  constexpr std::size_t n = std::tuple_size_v<std::remove_cvref_t<Tuple>>;

  for_each_tuple_element_impl(std::forward<F>(f), std::forward<Tuple>(t), std::make_index_sequence<n>());
}


} // end ubu::detail


#include "epilogue.hpp"

