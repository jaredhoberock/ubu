#pragma once

#include "../../detail/prologue.hpp"

#include <concepts>


namespace ubu::detail
{


template<class T>
concept number = (std::floating_point<T> or std::integral<T>);


template<class T>
concept not_a_number = !(number<T>);


} // end ubu::detail


#include "../../detail/epilogue.hpp"

