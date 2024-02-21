#pragma once

#include "../../../detail/prologue.hpp"

#include "coordinate.hpp"
#include "../traits/rank.hpp"
#include <tuple>
#include <type_traits>
#include <utility>

namespace ubu
{
namespace detail
{


// a forward declaration for is_subdimensional_recursive_impl
template<coordinate T1, coordinate T2>
constexpr bool is_subdimensional();

template<nonscalar_coordinate T1, nonscalar_coordinate T2, std::size_t... I>
constexpr bool is_subdimensional_recursive_impl(std::index_sequence<I...>)
{
  return (... and is_subdimensional<std::tuple_element_t<I,std::remove_cvref_t<T1>>, std::tuple_element_t<I,std::remove_cvref_t<T2>>>());
}

template<coordinate T1, coordinate T2>
constexpr bool is_subdimensional()
{
  if constexpr (scalar_coordinate<T1>)
  {
    // terminal case 1: a scalar subdimensional to anything, even another scalar
    return true;
  }
  else if constexpr (rank_v<T2> < rank_v<T1>)
  {
    // terminal case 2: T2 has lower rank than T1
    return false;
  }
  else
  {
    // recursive case: T1 has lower than or equal rank to T2
    return is_subdimensional_recursive_impl<T1,T2>(std::make_index_sequence<rank_v<T1>>{});
  }
}


} // end detail


// subdimensional is a recursive concept so it is implemented with a constexpr function
template<class T1, class T2>
concept subdimensional =
  coordinate<T1>
  and coordinate<T2>
  and detail::is_subdimensional<T1,T2>()
;


} // end ubu

#include "../../../detail/epilogue.hpp"

