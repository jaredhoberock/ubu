// note that this header file is special and does not use #pragma once

// This header #includes all header files underneath the detail/preprocessor directory.
// The only headers that should #include this file are detail/prologue.hpp and detail/epilogue.hpp
//
// A simple way to redefine configuration macros like ASPERA_NAMESPACE et al is to replace this
// header file with one containing custom definitions for all macros defined beneath the
// detail/config directory.

#include "preprocessor/annotation.hpp"
#include "preprocessor/exec_check_disable.hpp"
#include "preprocessor/has_cudart.hpp"
#include "preprocessor/has_exceptions.hpp"
#include "preprocessor/namespace.hpp"
#include "preprocessor/requires.hpp"

