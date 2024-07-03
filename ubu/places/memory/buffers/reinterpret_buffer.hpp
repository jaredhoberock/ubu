#pragma once

#include "../../../detail/prologue.hpp"
#include "../../../miscellaneous/integrals/size.hpp"
#include "../data.hpp"
#include "buffer_like.hpp"
#include <cstddef>
#include <span>

namespace ubu
{

// XXX reinterpret_buffer should probably return fancy_span

template<class T, buffer_like B>
constexpr std::span<T> reinterpret_buffer(B buffer, std::size_t num_objects)
{
  // XXX data returns pointer_like, so we probably need something more like to_raw_pointer
  return {reinterpret_cast<T*>(data(buffer)), num_objects};
}

template<class T, buffer_like B>
constexpr std::span<T> reinterpret_buffer(B buffer)
{
  return reinterpret_buffer<T>(buffer, size(buffer) / sizeof(T));
}

} // end ubu

#include "../../../detail/epilogue.hpp"

