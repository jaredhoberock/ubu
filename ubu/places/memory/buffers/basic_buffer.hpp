#pragma once

#include "../../../detail/prologue.hpp"
#include "../../../utilities/integrals/integral_like.hpp"
#include "../../../tensors/vectors/fancy_span.hpp"
#include <cstddef>

namespace ubu
{


template<integral_like S = std::size_t>
using basic_buffer = fancy_span<std::byte*,S>;


} // end ubu

#include "../../../detail/epilogue.hpp"

