#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/tensor_like.hpp"
#include "tensor_reference.hpp"
#include <memory>
#include <type_traits>

namespace ubu
{
namespace detail
{

// XXX this feels a bit hacky
template<tensor_like T>
using tensor_pointer_t = decltype(&std::declval<tensor_reference_t<T>&>());

} // end detail

// XXX using remove_cv_t seems like it may be inconsistent with the way std::pointer_traits<P> defines element_type
template<tensor_like T>
using tensor_element_t = std::remove_cv_t<typename std::pointer_traits<detail::tensor_pointer_t<T>>::element_type>;

} // end ubu

#include "../../detail/epilogue.hpp"

