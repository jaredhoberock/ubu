#pragma once

#include "../../../detail/prologue.hpp"

#include "../allocate.hpp"
#include "../concepts/allocator.hpp"
#include <memory>

namespace ubu
{
namespace detail
{

template<allocator A, class T>
struct allocator_pointer
{
  using type = allocate_result_t<T, std::remove_cvref_t<A>&,std::size_t>;
};

template<allocator A>
struct allocator_pointer<A,void>
{
  using type = typename std::allocator_traits<std::decay_t<A>>::pointer;
};

} // end detail


template<allocator A, class T = void>
using allocator_pointer_t = typename detail::allocator_pointer<A,T>::type;


} // end ubu

#include "../../../detail/epilogue.hpp"

