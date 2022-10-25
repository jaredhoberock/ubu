#pragma once

#include "../../detail/prologue.hpp"

#include <concepts>


namespace ubu::detail
{


template<class T>
concept number = (std::floating_point<T> or std::integral<T>);


template<class T>
concept not_number = (!number<T>);


template<class T1, class T2>
concept same_kind_of_number =
  number<T1>
  and number<T2>
  and ((std::integral<T1> and std::integral<T2>) or (std::floating_point<T1> and std::floating_point<T2>))
;


} // end ubu::detail


#include "../../detail/epilogue.hpp"

