#pragma once

#include "../../../detail/prologue.hpp"
#include "memory_view.hpp"
#include <cstddef>
#include <type_traits>

namespace ubu
{


template<class T>
concept buffer_like = memory_view_of<std::remove_cvref_t<T>,std::byte,std::size_t>;


} // end ubu

#include "../../../detail/epilogue.hpp"

