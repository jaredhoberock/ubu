#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "delete_after.hpp"
#include <type_traits>

ASPERA_NAMESPACE_OPEN_BRACE

template<class D>
concept asynchronous_deleter =
  requires{typename std::remove_cv_t<D>::event_type;} and
  event<typename std::remove_cv_t<D>::event_type> and
  requires(D d, const typename std::remove_cv_t<D>::event_type& e, typename std::remove_cv_t<D>::pointer ptr, std::size_t n)
  {
    {ASPERA_NAMESPACE::delete_after(d, e, ptr, n)} -> event;
  }
;

ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

