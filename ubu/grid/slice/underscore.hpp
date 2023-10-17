#pragma once

#include "../../detail/prologue.hpp"
#include <type_traits>

namespace ubu
{
namespace detail
{

struct underscore_t {};

template<class T>
constexpr bool is_underscore_v = std::is_same_v<std::remove_cvref_t<T>,underscore_t>;

} // end detail

constexpr detail::underscore_t _;

} // end ubu

#include "../../detail/epilogue.hpp"

