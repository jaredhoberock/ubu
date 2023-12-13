#pragma once

#include "../../detail/prologue.hpp"
#include "buffer_like.hpp"
#include <cstddef>
#include <ranges>

namespace ubu
{


// XXX WAR clang bug
#if defined(__clang_major__) and (__clang_major__ < 17)

struct empty_buffer : std::ranges::view_base
{
  static constexpr std::byte* begin() noexcept
  {
    return nullptr;
  }

  static constexpr std::byte* end() noexcept
  {
    return nullptr;
  }

  static constexpr std::byte* data() noexcept
  {
    return nullptr;
  }

  static constexpr int size() noexcept
  {
    return 0;
  }

  static constexpr bool empty() noexcept
  {
    return true;
  }
};

static_assert(buffer_like<empty_buffer>);

#else
using empty_buffer = std::ranges::empty_view<std::byte>;
#endif


template<class T>
concept nonempty_buffer_like =
  buffer_like<T>
  and not std::same_as<std::remove_cvref_t<T>, empty_buffer>
;


} // end ubu

#include "../../detail/epilogue.hpp"

