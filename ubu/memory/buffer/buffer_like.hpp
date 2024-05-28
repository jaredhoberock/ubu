#pragma once

#include "../../detail/prologue.hpp"
#include <cstddef>
#include <ranges>
#include <type_traits>
#include <utility>

namespace ubu
{


// XXX we should make buffer_like a refinement of span_like
template<class T>
concept buffer_like =
  std::ranges::view<std::remove_cvref_t<T>>
  and std::ranges::contiguous_range<T>
  and std::ranges::sized_range<T>
  and std::same_as<std::ranges::range_value_t<T>, std::byte>
;


} // end ubu

#include "../../detail/epilogue.hpp"

