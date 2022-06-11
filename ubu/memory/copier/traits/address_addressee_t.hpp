#pragma once

#include "../../../detail/prologue.hpp"

#include "../address.hpp"

namespace ubu
{

template<class A>
  requires address<A>
using address_addressee_t = detail::address_addressee_t<A>;

} // end ubu

#include "../../../detail/epilogue.hpp"

