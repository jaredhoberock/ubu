#pragma once

#include "../../../detail/prologue.hpp"

#include "../concepts/coordinate.hpp"
#include <type_traits>


namespace ubu
{

// XXX this definition assumes that brace-initializing T will actually put a 0 in each mode of T
template<coordinate T>
constexpr std::remove_cvref_t<T> zeros_v{};

template<coordinate T>
using zeros_t = std::remove_cvref_t<decltype(zeros_v<T>)>;

} // end ubu

#include "../../../detail/epilogue.hpp"

