#pragma once

#include "../../../detail/prologue.hpp"

#include "../../thread_scope.hpp"
#include "../concepts/cooperator.hpp"
#include <string_view>

namespace ubu
{

template<cooperator C>
  requires detail::has_static_thread_scope<C>
inline constexpr const std::string_view cooperator_thread_scope_v = thread_scope.operator()<C>();

} // end ubu

#include "../../../detail/epilogue.hpp"

