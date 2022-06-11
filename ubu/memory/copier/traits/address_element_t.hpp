#pragma once

#include "../../../detail/prologue.hpp"

#include "../address.hpp"

namespace ubu
{

template<class A>
  requires address<A>
using address_element_t = detail::address_element_t<A>;

} // end ubu

#include "../../../detail/epilogue.hpp"


