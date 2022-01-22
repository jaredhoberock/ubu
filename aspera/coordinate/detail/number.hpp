#pragma once

#include "../../detail/prologue.hpp"

#include <concepts>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
concept number = (std::floating_point<T> or std::integral<T>);


} // end detail


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"
