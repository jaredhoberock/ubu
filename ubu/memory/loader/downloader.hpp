#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/initial_happening.hpp"
#include "address.hpp"
#include "download_after.hpp"
#include <concepts>
#include <type_traits>

namespace ubu
{


template<class D>
concept downloader =
  std::is_nothrow_move_constructible_v<std::remove_cvref_t<D>>
  and std::is_nothrow_destructible_v<std::remove_cvref_t<D>>
  and std::equality_comparable<D>
  and address<typename std::remove_cvref_t<D>::address_type>
  and requires(D d)
  {
    initial_happening(d);
  }
  and requires(D d, initial_happening_result_t<D> before, typename std::remove_cvref_t<D>::address_type from, std::size_t num_bytes, void* to)
  {
    download_after(d, before, from, num_bytes, to);
  }
;


} // end ubu

#include "../../detail/epilogue.hpp"

