#pragma once

#include "../../detail/prologue.hpp"
#include "buffer_like.hpp"
#include <cstddef>
#include <ranges>
#include <span>

namespace ubu
{

template<class T, buffer_like B>
constexpr std::span<T> reinterpret_buffer(B buffer, std::size_t num_objects)
{
  return {reinterpret_cast<T*>(std::ranges::data(buffer)), num_objects};
}

template<class T, buffer_like B>
constexpr std::span<T> reinterpret_buffer(B buffer)
{
  return reinterpret_buffer<T>(buffer, std::ranges::size(buffer) / sizeof(T));
}

} // end ubu

#include "../../detail/epilogue.hpp"

