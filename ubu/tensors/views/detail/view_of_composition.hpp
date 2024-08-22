#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinates/concepts/congruent.hpp"
#include "../../coordinates/element.hpp"
#include "../../concepts/view.hpp"
#include "../../concepts/composable.hpp"
#include "../../shapes/shape.hpp"
#include "../../traits/tensor_element.hpp"
#include <concepts>

namespace ubu::detail
{


template<class R, class A, class B>
concept view_of_composition =
  view<R>
  and composable<A,B>
  and congruent<shape_t<R>, shape_t<B>>
  and std::same_as<tensor_element_t<R>, element_t<A, tensor_element_t<B>>>
;


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

