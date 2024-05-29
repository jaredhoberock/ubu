#pragma once

#include "../../detail/prologue.hpp"
#include "../../miscellaneous/constant.hpp"
#include <tuple>
#include <utility>

namespace ubu::detail
{


template<std::size_t... I>
constexpr std::tuple<constant<I>...> as_tuple_of_constants(std::index_sequence<I...>)
{
  return {};
}


} // end ubu::detail

#include "../../detail/epilogue.hpp"

