#pragma once

#include "../detail/prologue.hpp"

#include "wait.hpp"


ASPERA_NAMESPACE_OPEN_BRACE


template<class E>
concept event = requires(E e) { ASPERA_NAMESPACE::wait(e); };


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

