#pragma once

#include "../detail/prologue.hpp"
#include "concepts/elemental_invocable.hpp"
#include "concepts/tensor_like.hpp"
#include "coordinates/element.hpp"
#include "element_exists.hpp"
#include "shapes/shape.hpp"
#include "shapes/shape_size.hpp"
#include "traits/tensor_element.hpp"
#include "vectors/vector_like.hpp"
#include <concepts>
#include <optional>

namespace ubu
{


template<tensor_like T>
constexpr bool empty(T&& tensor)
{
  return ubu::shape_size(ubu::shape(tensor)) == 0;
}


// XXX TODO: to generalize this to tensor_like, we need to use bulk_execute + inline_executor
//           to generate coordinates in an efficient (i.e., loop-unrollable) way
// XXX TODO: receive an optional init parameter
// XXX TODO: require that elemental_invoke_result_t is convertible to tensor_element_t<V>
template<vector_like V, elemental_invocable<V,V> F>
constexpr std::optional<tensor_element_t<V>> reduce(V&& vector, F binary_op)
{
  std::optional<tensor_element_t<V>> result;

  bool found_first_element = false;

  for(int i = 0; i < shape(vector); ++i)
  {
    if(element_exists(vector, i))
    {
      if(found_first_element)
      {
        *result = binary_op(*result, element(vector, i));
      }
      else
      {
        result = element(vector, i);
        found_first_element = true;
      }
    }
  }

  return result;
}


} // end ubu

#include "../detail/epilogue.hpp"

