#pragma once

#include "../../../detail/prologue.hpp"

#include "slice.hpp"
#include <utility>

namespace ubu
{

template<class T, class K>
concept sliceable = requires(T t, K k)
{
  slice(std::forward<T>(t), std::forward<K>(k));
};

} // end ubu


#include "../../../detail/epilogue.hpp"

