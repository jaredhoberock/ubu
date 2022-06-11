#pragma once

#include "../../detail/prologue.hpp"

#include "address_difference.hpp"
#include "advance_address.hpp"
#include "make_null_address.hpp"
#include <concepts>
#include <cstdint>

namespace ubu
{

namespace detail
{


template<class T>
struct address_addressee
{
  using type = typename T::addressee_type;
};

template<class T>
struct address_addressee<T*>
{
  using type = T;
};

template<class T>
using address_addressee_t = typename address_addressee<T>::type;


} // end detail


template<class A>
concept address = 
  std::regular<A>
  and std::totally_ordered<A>

  and requires
  {
    typename detail::address_addressee_t<std::remove_cvref_t<A>>;
    make_null_address<A>;
  }
;


template<class A, class T>
concept address_for = address<A> and std::same_as<detail::address_addressee_t<A>, T>;


template<class A>
concept typed_address =
  address<A> 
  and !std::is_void_v<detail::address_addressee_t<std::remove_cvref_t<A>>>
  and requires(A a, A b, std::ptrdiff_t n)
  {
    advance_address(a, n);
    { address_difference(a, b) } -> std::convertible_to<std::ptrdiff_t>;
  }
;


} // end ubu

#include "../../detail/epilogue.hpp"

