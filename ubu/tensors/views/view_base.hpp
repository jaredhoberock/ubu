#pragma once

#include "../../detail/prologue.hpp"
#include <ranges>

namespace ubu
{


// just use std::ranges::view_base until there's a reason not to
using view_base = std::ranges::view_base;

// allows a type to conditionally be a view
template<bool condition>
struct view_base_if {};

template<>
struct view_base_if<true> : view_base {};


} // end ubu

#include "../../detail/epilogue.hpp"

