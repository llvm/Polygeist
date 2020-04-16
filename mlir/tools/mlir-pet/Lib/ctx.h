#ifndef CTX_H
#define CTX_H

#include "pet.h"
#include "isl/isl-noexceptions.h"

namespace util {

/// Simple wrapper around isl::ctx that allocates a new context during
/// construction and frees it during destruction, e.g. when a stack-allocated
/// instance of ScopedCtx goes out of scope.
/// Implicitly convertible to both isl_ctx* and isl::ctx for convenience.
/// Intentionally not copy-constructible or copy-assignable as it would have
/// required reference counting.  Move-constructible to enable ownership
/// transfer.
class ScopedCtx {
public:
  ScopedCtx() : ctx(isl_ctx_alloc_with_pet_options()) {}
  explicit ScopedCtx(isl::ctx &&ctx) : ctx(ctx) {}
  ScopedCtx(const ScopedCtx &) = delete;
  ScopedCtx(ScopedCtx &&) = default;
  ~ScopedCtx() { isl_ctx_free(ctx.release()); }

  ScopedCtx &operator=(const ScopedCtx &) = delete;
  ScopedCtx &operator=(ScopedCtx &&) = default;

  operator isl::ctx() { return ctx; }
  operator isl_ctx *() { return ctx.get(); }

private:
  isl::ctx ctx;
};

} // namespace util

#endif
