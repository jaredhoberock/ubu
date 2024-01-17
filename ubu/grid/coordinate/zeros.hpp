#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/for_each_tuple_element.hpp"
#include "concepts/coordinate.hpp"
#include <type_traits>


namespace ubu
{

// XXX this definition assumes that brace-initializing T will actually put a 0 in each mode of T
template<coordinate T>
constexpr std::remove_cvref_t<T> zeros{};

} // end ubu

#include "../../detail/epilogue.hpp"

