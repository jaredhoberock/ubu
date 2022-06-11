#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "address.hpp"
#include "copy_n.hpp"
#include <concepts>
#include <type_traits>

namespace ubu
{


template<class C, class From, class To>
concept copier_between =
  std::is_nothrow_move_constructible_v<std::remove_cvref_t<C>>
  and std::is_nothrow_destructible_v<std::remove_cvref_t<C>>
  and std::equality_comparable<C>

  // From and To are addresses
  and address<From>
  and address<To>

  // a copier must copy elements from From to To
  and requires(C c, From from, std::size_t n, To to)
  {
    ubu::copy_n(c, from, n, to);
  }
;


template<class C, class T>
concept copier_of =
  // T shouldn't be a reference
  !std::is_reference_v<T>

  // a copier must be able to instantiate an address for Ts
  and requires
  {
    typename std::remove_cvref_t<C>::template address<T>;
  }

  // if T is not void, then a copier needs to be able to copy Ts
  and (std::is_void_v<T> or (

    // if T is copy assignable,
    // a copier must copy from const T* to its native address
    (!std::is_copy_assignable_v<T> or copier_between<C, const T*, typename std::remove_cvref_t<C>::template address<T>>)

    // a copier must copy from its native address to T*
    and copier_between<C, typename std::remove_cvref_t<C>::template address<T>, std::remove_cvref_t<T>*>

    // if T is copy assignable,
    // a copier must copy bytes from native address to native address
    and (!std::is_copy_assignable_v<T> or copier_between<C, typename std::remove_cvref_t<C>::template address<T>, typename std::remove_cvref_t<C>::template address<T>>)
  ))
;


} // end ubu

#include "../../detail/epilogue.hpp"

