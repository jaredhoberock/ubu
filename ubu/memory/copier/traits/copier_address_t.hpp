#pragma once

#include "../../../detail/prologue.hpp"

#include "../copier.hpp"

namespace ubu
{

template<class C, class T>
  requires copier_of<C,T>
using copier_address_t = typename std::remove_cvref_t<C>::template address<T>;

} // end ubu

#include "../../../detail/epilogue.hpp"

