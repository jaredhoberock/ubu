#pragma once

#include "../detail/prologue.hpp"

#include <type_traits>

namespace ubu
{


template<class T>
concept plain_old_data = std::is_standard_layout_v<T> and std::is_trivial_v<T>;


template<class T>
concept plain_old_data_or_void = (plain_old_data<T> or std::is_void_v<T>);


} // end ubu

#include "../detail/epilogue.hpp"

