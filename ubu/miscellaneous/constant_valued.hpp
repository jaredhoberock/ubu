#pragma once

#include "../detail/prologue.hpp"
#include "../tensor/coordinate/detail/tuple_algorithm.hpp"
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ubu
{
namespace detail
{

template<class T>
concept scalar_constant =
  not tuple_like<T>
  and requires()
  {
    std::remove_cvref_t<T>::value;

    requires std::is_object_v<decltype(std::remove_cvref_t<T>::value)>;
  }
;

template<class T>
struct is_nonscalar_constant : std::false_type {};

template<tuple_like T, std::size_t... I>
constexpr bool tuple_elements_are_constants(std::index_sequence<I...>)
{
  // this checks whether each element of T is either scalar_constant or is_nonscalar_constant

  return (... and (scalar_constant<std::tuple_element_t<I,std::remove_cvref_t<T>>> or is_nonscalar_constant<std::tuple_element_t<I,std::remove_cvref_t<T>>>::value));
}

template<tuple_like T>
struct is_nonscalar_constant<T>
{
  constexpr static bool value = tuple_elements_are_constants<T>(tuple_indices<T>);
};


// because nonscalar_constant is recursive, it is partially implemented with a type trait
template<class T>
concept nonscalar_constant =
  tuple_like<T>
  and is_nonscalar_constant<T>::value;
;


template<class T>
struct dispatch_get_constant_value
{
  template<class U = std::remove_cvref_t<T>>
    requires scalar_constant<U>
  constexpr auto operator()() const
  {
    return U::value;
  }

  template<class U = std::remove_cvref_t<T>>
    requires (not scalar_constant<U>
              and nonscalar_constant<U>)
  constexpr auto operator()() const
  {
    // recurse across the elements of the tuple U
    // XXX this assumes U is default constructible
    return tuple_zip_with(U(), [](const auto& element)
    {
      return dispatch_get_constant_value<decltype(element)>();
    });
  }
};


} // end detail

template<class T>
inline constexpr detail::dispatch_get_constant_value<T> get_constant_value;

template<class T>
concept constant_valued =
  requires()
  {
    get_constant_value<T>();
  }
;

template<constant_valued T>
inline constexpr auto constant_value_v = get_constant_value<T>();

template<constant_valued T>
using constant_value_t = decltype(constant_value_v<T>);

} // end ubu

#include "../detail/epilogue.hpp"

