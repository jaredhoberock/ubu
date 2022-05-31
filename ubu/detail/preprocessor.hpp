// note that this header file is special and does not use #pragma once

// This header #includes all header files underneath the detail/preprocessor directory.
// The only headers that should #include this file are detail/prologue.hpp and detail/epilogue.hpp
//
// A simple way to redefine configuration macros like UBU_NAMESPACE et al is to replace this
// header file with one containing custom definitions for all macros defined beneath the
// detail/config directory.

#include "preprocessor/namespace.hpp"
#include "preprocessor/target.hpp"

