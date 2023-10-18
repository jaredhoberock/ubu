// note that this header file is special and does not use #pragma once

// __is_trivially_relocatable expands to an approximation of the built-in trait when a\
// compiler does not have the __is_trivially_copyable builtin
// otherwise, __is_trivially_relocatable is not a preprocessor symbol (because the compiler already supports the built-in)

#if !defined(UBU_IS_TRIVIALLY_RELOCATABLE_NEEDS_UNDEF)

#if !__has_builtin(__is_trivially_relocatable)
#define __is_trivially_relocatable(T) __is_trivially_copyable(T)
#define UBU_IS_TRIVIALLY_RELOCATABLE_NEEDS_UNDEF
#endif

#elif defined(UBU_IS_TRIVIALLY_RELOCATABLE_NEEDS_UNDEF)

#undef __is_trivially_relocatable
#undef UBU_IS_TRIVIALLY_RELOCATABLE_NEEDS_UNDEF

#endif

