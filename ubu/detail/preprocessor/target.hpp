// note that this header file is special and does not use #pragma once

// UBU_TARGET expands to target when compiled by circle
// otherwise, constexpr
// it is used to write if target(...) branches portably

#if !defined(UBU_TARGET)

#  ifdef __circle_lang__
#    define UBU_TARGET target
#  else
#    define UBU_TARGET constexpr
#  endif
#  define UBU_TARGET_NEEDS_UNDEF

#elif defined(UBU_TARGET_NEEDS_UNDEF)

#undef UBU_TARGET
#undef UBU_TARGET_NEEDS_UNDEF

#endif

