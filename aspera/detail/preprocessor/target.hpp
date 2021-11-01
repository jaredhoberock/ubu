// note that this header file is special and does not use #pragma once

// ASPERA_TARGET expands to target when compiled by circle
// otherwise, constexpr
// it is used to write if target(...) branches portably

#if !defined(ASPERA_TARGET)

#  ifdef __circle_lang__
#    define ASPERA_TARGET target
#  else
#    define ASPERA_TARGET constexpr
#  endif
#  define ASPERA_TARGET_NEEDS_UNDEF

#elif defined(ASPERA_TARGET_NEEDS_UNDEF)

#undef ASPERA_TARGET
#undef ASPERA_TARGET_NEEDS_UNDEF

#endif

