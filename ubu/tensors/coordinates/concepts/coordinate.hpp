#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../utilities/tuples.hpp"
#include "../detail/to_integral_like.hpp"
#include "../element.hpp"
#include "../traits/rank.hpp"
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>


namespace ubu
{
namespace detail
{


template<class T>
constexpr bool is_coordinate()
{
  if constexpr (integral_like<T>)
  {
    return true;
  }
  else if constexpr (tuples::tuple_like<T>)
  {
    // note that this case handles both unit_like and tuples of a single integral_like

    auto all_elements_are_coordinates = []<std::size_t... I>(std::index_sequence<I...>)
    {
      return (... and is_coordinate<tuples::element_t<I,T>>());
    };

    return all_elements_are_coordinates(tuples::indices_v<T>);
  }
  else
  {
    return false;
  }
}


} // end detail


template<class T>
concept coordinate = detail::is_coordinate<T>();


template<class T>
concept nullary_coordinate =
  coordinate<T>
  and rank_v<T> == 0
;


template<class T>
concept unary_coordinate =
  coordinate<T>
  and rank_v<T> == 1
;

// XXX eliminate this name eventually
template<class T>
concept scalar_coordinate = unary_coordinate<T>;


template<class T>
concept multiary_coordinate = 
  coordinate<T>
  and (rank_v<T> > 1)
;

// XXX eliminate this name eventually
template<class T>
concept nonscalar_coordinate = multiary_coordinate<T>;


template<class... Types>
concept coordinates = (... and coordinate<Types>);

template<class T, std::size_t N>
concept coordinate_of_rank = coordinate<T> and (rank_v<T> == N);

template<class... Types>
concept nonscalar_coordinates = (... and nonscalar_coordinate<Types>);


// XXX consider reorganizing coordinate_for and element underneath tensor
template<class C, class T>
concept coordinate_for =
  coordinate<C>
  and requires(C coord, T obj)
  {
    ubu::element(obj, coord);
  }
;


} // end ubu


#include "../../../detail/epilogue.hpp"

