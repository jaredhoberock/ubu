#pragma once

#include "../../detail/prologue.hpp"

#include "../traits/tensor_element.hpp"
#include "../traits/tensor_shape.hpp"
#include "view.hpp"
#include <type_traits>

namespace ubu
{

template<class T, class E, class S = void>
concept view_of =
  view<T>
  and std::is_object_v<E>
  and std::same_as<tensor_element_t<T>,E>
  and (std::is_void_v<S> or (coordinate<S> and congruent<S,tensor_shape_t<T>>))
;

} // end ubu

#include "../../detail/epilogue.hpp"

