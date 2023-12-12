#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/initial_happening.hpp"
#include "address.hpp"
#include "upload_after.hpp"
#include <concepts>
#include <type_traits>

namespace ubu
{


template<class U>
concept uploader =
  std::is_nothrow_move_constructible_v<std::remove_cvref_t<U>>
  and std::is_nothrow_destructible_v<std::remove_cvref_t<U>>
  and std::equality_comparable<U>
  and address<typename std::remove_cvref_t<U>::address_type>
  and requires(U u)
  {
    initial_happening(u);
  }
  and requires(U u, initial_happening_result_t<U> before, const void* from, std::size_t num_bytes, typename std::remove_cvref_t<U>::address_type to)
  {
    upload_after(u, before, from, num_bytes, to);
  }
;


} // end ubu

#include "../../detail/epilogue.hpp"

