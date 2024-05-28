#pragma once

#include "../../detail/prologue.hpp"
#include "buffer_like.hpp"
#include <cstddef>
#include <ranges>

namespace ubu
{


using empty_buffer = std::ranges::empty_view<std::byte>;


template<class T>
concept nonempty_buffer_like =
  buffer_like<T>
  and not std::same_as<std::remove_cvref_t<T>, empty_buffer>
;


} // end ubu

#include "../../detail/epilogue.hpp"

