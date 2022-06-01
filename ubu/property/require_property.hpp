#pragma once

#include "../detail/prologue.hpp"

#include "mix.hpp"
#include "property.hpp"
#include <concepts>
#include <type_traits>
#include <utility>


namespace ubu
{

namespace detail
{


template<class T, class P>
concept has_mutable_property =
  has_property<T,P>
  and requires(T t, P prop)
  {
    prop(t, prop.value);
  }
;


} // end detail


// T doesnt have the property but can mix it in
template<class P, can_mix_property<P> T>
auto require_property(T&& object, P prop)
{
  return mix(std::forward<T>(object), prop);
}


// T has the property
template<class P, detail::has_mutable_property<P> T>
  requires std::copyable<std::remove_cvref_t<T>>
std::remove_cvref_t<T> require_property(T&& object, P prop)
{
  // if the property's value is already equal to what is required, just return the object
  if(prop(std::forward<T>(object)) == prop.value)
  {
    return std::forward<T>(object);
  }

  // make a copy of the input object
  std::remove_cvref_t<T> result = std::forward<T>(object);

  // mutate the copy with the value of the property of interest
  prop(result, prop.value);

  // return the copy
  return result;
}


} // end ubu

#include "../detail/epilogue.hpp"

