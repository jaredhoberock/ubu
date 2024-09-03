#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/tensor.hpp"
#include "tensor_reference.hpp"
#include <memory>
#include <type_traits>

namespace ubu
{
namespace detail
{

// XXX this hacky business of forming a pointer and then using std::pointer_traits is intended to deal with fancy reference types
template<tensor T>
using tensor_pointer_t = decltype(&std::declval<tensor_reference_t<T>&>());

template<tensor T>
struct tensor_element
{
  // XXX using remove_cv_t seems like it may be inconsistent with the way std::pointer_traits<P> defines element_type
  using type = std::remove_cv_t<typename std::pointer_traits<tensor_pointer_t<T>>::element_type>;
};

// this is a special case for tensors that "return" void; their element type is void
template<tensor T>
  requires std::is_void_v<tensor_reference_t<T>>
struct tensor_element<T>
{
  using type = void;
};

} // end detail


template<tensor T>
using tensor_element_t = typename detail::tensor_element<T>::type;

} // end ubu

#include "../../detail/epilogue.hpp"

