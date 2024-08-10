#pragma once

#include "../../../detail/prologue.hpp"
#include "memory_view.hpp"
#include <cstddef>

namespace ubu
{


template<class T>
concept buffer_like = memory_view_of<T,std::byte,std::size_t>;


} // end ubu

#include "../../../detail/epilogue.hpp"

