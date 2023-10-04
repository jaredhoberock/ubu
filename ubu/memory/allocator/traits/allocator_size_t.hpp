#pragma once

#include "../../../detail/prologue.hpp"

#include "../concepts/allocator.hpp"
#include <memory>

namespace ubu
{


template<allocator A>
using allocator_size_t = typename std::allocator_traits<std::decay_t<A>>::size_type;


} // end ubu

#include "../../../detail/epilogue.hpp"

