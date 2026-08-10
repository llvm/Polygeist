# Issues in RaiseToLinalg.cpp

This document tracks known issues and limitations in the `RaiseToLinalg.cpp` pass.

---

## Issue #1: Strided Memory Access Pattern Conversion Failure

**Date Identified:** October 15, 2025

**Status:** 🔴 Open

### Description

The `RaiseToLinalg` pass fails when converting affine loops with strided memory access patterns (e.g., `x[i*2]`) to `linalg.generic` operations. The pass creates invalid MLIR that fails verification.

### Error Message

```
error: 'linalg.generic' op expected the shape-to-loops map to be non-null
```

### Root Cause

When the pass encounters strided access patterns like:
```mlir
affine.for %arg0 = 0 to 3 {
  %1 = affine.load %x[%arg0 * 2] : memref<6xf64>
  %2 = affine.load %y[%arg0 * 2] : memref<6xf64>
  // ... operations ...
  affine.store %result, %y[%arg0 * 2] : memref<6xf64>
}
```

The pass generates:
```mlir
linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0 * 2)>, affine_map<(d0) -> (d0 * 2)>],
  iterator_types = [#linalg.iterator_type<parallel>]
} ins(%x : memref<6xf64>) outs(%y : memref<6xf64>) { ... }
```

**Problem:** `linalg.generic` with non-strided memref types doesn't support non-identity indexing maps with multiplicative stride patterns. This causes a verification failure because the shape-to-loops mapping becomes ambiguous.

### Example Test Case

From `blas/daxpy.c` with stride=2:
```c
// Computing: y[::2] += alpha * x[::2]  (every other element)
daxpy(3, 10.0, x, 2, y, 2);  // incx=2, incy=2
```

This generates the failing MLIR pattern.

### Expected Behavior

The pass should create strided memory views using `polygeist.submap` before the `linalg.generic`, similar to how the non-strided version works:

```mlir
%strided_x = polygeist.submap(%x, %stride, %size) <{map = affine_map<(d0)[s0] -> (d0 * s0)>}>
%strided_y = polygeist.submap(%y, %stride, %size) <{map = affine_map<(d0)[s0] -> (d0 * s0)>}>

linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],  // Identity maps!
  iterator_types = [#linalg.iterator_type<parallel>]
} ins(%strided_x : memref<?xf64>) outs(%strided_y : memref<?xf64>) { ... }
```

### Workaround

Currently, strided BLAS operations (with `incx != 1` or `incy != 1`) fail when passed through the `--raise-affine-to-linalg-pipeline`. 

**Temporary workaround:**
- Only test with `simple_*` versions that have contiguous memory access
- Manually avoid stride parameters in BLAS kernels during testing

### Proposed Fix

Modify the `AffineForOpRaising::matchAndRewrite` function in `RaiseToLinalg.cpp`:

1. **Detect non-identity affine maps** in memory access patterns
2. **Extract stride information** from expressions like `d0 * c` where `c` is a constant
3. **Create `polygeist.submap` operations** to materialize strided views
4. **Generate identity indexing maps** for `linalg.generic`
5. **Adjust loop bounds** to reflect the strided iteration space

### Impact

**Affected Operations:**
- All Level 1 BLAS with stride parameters: `daxpy`, `ddot`, `dscal`, `dcopy`, `dasum`, `dnrm2`
- Any custom kernels with strided memory access patterns

**Severity:** High - Blocks usage of standard BLAS interfaces with stride support

### Related Files

- `lib/polygeist/Passes/RaiseToLinalg.cpp` - Main pass implementation
- `blas/daxpy.c` - Test case demonstrating the issue
- Debug log with failure: See commit around Oct 15, 2025

### References

- MLIR Linalg Dialect Documentation: https://mlir.llvm.org/docs/Dialects/Linalg/
- Polygeist Documentation on memory views
- Debug output showing failure: `/home/arjaiswal/Polygeist/out` (lines 2579-2835)

---

## Issue #2: Reduction Loops Not Supported

**Date Identified:** October 15, 2025

**Status:** 🔴 Open

### Description

The `AffineForOpRaising` pattern immediately rejects any `affine.for` loop that has results (uses `iter_args`), preventing reduction operations from being raised to linalg. This blocks operations like `dasum`, `ddot`, and `dnrm2` that accumulate values.

### Error Pattern

```
REJECTED: Loop has results
```

### Root Cause

In `RaiseToLinalg.cpp`, the `AffineForOpRaising::matchAndRewrite` function contains:

```cpp
// Early rejection if loop has results
if (op.getNumResults() > 0) {
  LLVM_DEBUG(llvm::dbgs() << "\nREJECTED: Loop has results\n");
  return failure();
}
```

This check was designed for simple parallel loops but prevents handling of reduction patterns.

### Example Test Case

From `blas/dasum.c`:
```c
double simple_dasum(int n, const double* x, int incx) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += fabs(x[i * incx]);  // Reduction: accumulate sum
    }
    return sum;
}
```

Generated MLIR (correctly represents reduction):
```mlir
%sum = affine.for %arg0 = 0 to %n iter_args(%arg1 = %cst) -> (f64) {
  %val = affine.load %x[%arg0] : memref<?xf64>
  %abs = math.absf %val : f64
  %new_sum = arith.addf %arg1, %abs : f64
  affine.yield %new_sum : f64
}
```

This loop is **rejected** because it has results.

### Expected Behavior

The pass should detect reduction patterns and generate appropriate linalg operations:

```mlir
%sum = linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
  iterator_types = [#linalg.iterator_type<reduction>]
} ins(%x : memref<?xf64>) outs(%init : memref<f64>) {
^bb0(%in: f64, %out: f64):
  %abs = math.absf %in : f64
  %sum = arith.addf %out, %abs : f64
  linalg.yield %sum : f64
}
```

Or use `linalg.reduce` operation for simple reductions.

### Proposed Fix

1. **Remove or conditionally apply** the early rejection for loops with results
2. **Add reduction pattern detection**: Check if loop body accumulates into `iter_args`
3. **Identify reduction operation type**: `addf`, `mulf`, `minf`, `maxf`, etc.
4. **Generate linalg with reduction semantics**: Use `linalg.iterator_type<reduction>` and proper output handling
5. **Handle scalar results**: Map scalar reductions to 0-D tensors or memrefs

### Impact

**Affected Operations:**
- **Level 1 BLAS reductions**: `ddot` (dot product), `dasum` (sum of absolute values), `dnrm2` (Euclidean norm)
- Any custom kernels with accumulation/reduction patterns

**Severity:** High - Completely blocks reduction operations, which are core BLAS primitives

### Workaround

None currently available. Reduction operations cannot be raised to linalg with the current pass implementation.

### Related Files

- `lib/polygeist/Passes/RaiseToLinalg.cpp` - Lines with `getNumResults()` check
- `blas/dasum.c`, `blas/ddot.c`, `blas/dnrm2.c` - Test cases
- Debug output: `/home/arjaiswal/Polygeist/out` (lines 2902-2912, 3080-3090)

---

## Issue #3: LinalgDebufferize Cannot Handle polygeist.submap Operations

**Date Identified:** October 17, 2025

**Status:** 🔴 Open

### Description

The `LinalgDebufferize` pass fails to transform `linalg.generic` operations that consume `polygeist.submap` results. The pass only recognizes `memref.AllocaOp`, `memref.AllocOp`, and function arguments as "roots" for debufferization, causing it to skip `polygeist.submap` operations entirely. This means any `linalg.generic` that operates on strided views created by `polygeist.submap` remains in memref form and is never debufferized.

### Error Pattern

Debug logs show:
```
[User 0] Processing: polygeist.submap
  Unknown user type (skipping): polygeist.submap
```

No error is thrown, but the pass silently skips the operations, leaving them untransformed.

### Root Cause

In `LinalgDebufferize.cpp`, the `handleMemref` lambda only processes memrefs from specific sources:
```cpp
for (auto arg : funcOp.getArguments()) {
  if (failed(handleMemref(arg))) return failure();
}
// Also handles memref.alloc and memref.alloca
```

When `polygeist.submap` creates a view:
```mlir
%strided_x = polygeist.submap(%arg2, %stride, %size) 
             <{map = affine_map<(d0)[s0] -> (d0 * s0)>}>
             : (memref<?xf32>, index, index) -> memref<?xf32>

linalg.generic ins(%strided_x : memref<?xf32>) ...
```

The pass sees `%strided_x` as a user of the base memref, classifies it as "Unknown user type", and skips it. Consequently, the `linalg.generic` that uses `%strided_x` is never encountered or transformed.

### Example Test Case

From `saxpy_linalg.mlir` (after RaiseToLinalg):
```mlir
func.func @saxpy(%n: i32, %alpha: f32, %x: memref<?xf32>, %incx: i32, 
                 %y: memref<?xf32>, %incy: i32) {
  %stride_x = arith.index_cast %incx : i32 to index
  %stride_y = arith.index_cast %incy : i32 to index
  %size = arith.index_cast %n : i32 to index
  
  // Create strided views
  %x_view = polygeist.submap(%x, %stride_x, %size) 
            <{map = affine_map<(d0)[s0] -> (d0 * s0)>}>
            : (memref<?xf32>, index, index) -> memref<?xf32>
  
  %y_view = polygeist.submap(%y, %stride_y, %size) 
            <{map = affine_map<(d0)[s0] -> (d0 * s0)>}>
            : (memref<?xf32>, index, index) -> memref<?xf32>
  
  // This linalg.generic is NEVER debufferized!
  linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%x_view : memref<?xf32>) outs(%y_view : memref<?xf32>) {
  ^bb0(%x_val: f32, %y_val: f32):
    %mul = arith.mulf %alpha, %x_val : f32
    %add = arith.addf %y_val, %mul : f32
    linalg.yield %add : f32
  }
  return
}
```

Running `polygeist-opt --linalg-debufferize` leaves this IR unchanged.

### Why Can't We Use linalg.generic for Materialization?

Initial attempts to materialize `polygeist.submap` as `linalg.generic` operations fail because:

**MLIR Constraint:** `linalg.generic` indexing maps **cannot contain symbols** - only dimensions are allowed.

```mlir
// ❌ INVALID - symbols not allowed in linalg.generic indexing maps
linalg.generic {
  indexing_maps = [
    affine_map<(d0)[s0] -> (d0 * s0)>,  // Symbol [s0] causes error!
    affine_map<(d0) -> (d0)>
  ]
} ins(%base : tensor<?xf32>) outs(%view : tensor<?xf32>) { ... }
```

**Error message:**
```
error: 'linalg.generic' op unexpected symbols in indexing_map #0
```

This is a fundamental constraint of the Linalg dialect - iteration spaces must be determined solely by operand shapes (dimensions), not by runtime parameters (symbols).

### Proposed Solutions

#### **Solution 1: SCF Loops with tensor.extract/insert (Most General)**

Transform `polygeist.submap` into explicit gather/scatter loops:

```mlir
// For: polygeist.submap(%base, %stride, %size) with map (d0)[s0] -> (d0 * s0)

%base_tensor = bufferization.to_tensor %base : memref<?xf32>
%view_init = tensor.empty(%size) : tensor<?xf32>

%view = scf.for %i = %c0 to %size step %c1 
        iter_args(%acc = %view_init) -> (tensor<?xf32>) {
  // Compute strided index: i * stride
  %strided_idx = arith.muli %i, %stride : index
  
  // Gather: extract from strided position in base
  %val = tensor.extract %base_tensor[%strided_idx] : tensor<?xf32>
  
  // Insert into contiguous position in view
  %updated = tensor.insert %val into %acc[%i] : tensor<?xf32>
  scf.yield %updated : tensor<?xf32>
}
```

**Pros:**
- ✅ Works for **dynamic strides** (runtime values)
- ✅ Always valid MLIR
- ✅ Can be fused with subsequent operations by SCF loop fusion
- ✅ After fusion + bufferization, produces optimal code

**Cons:**
- ❌ Not a single linalg operation (harder for linalg-specific transformations)
- ❌ Introduces loop overhead for materialization (eliminated by fusion)

#### **Solution 2: tensor.extract_slice for Static Strides (Optimized)**

For compile-time constant strides, use native tensor view operations:

```mlir
// If stride is known at compile-time (e.g., constant 2)
%view = tensor.extract_slice %base_tensor[0][%size][2] : 
        tensor<?xf32> to tensor<?xf32>
```

**Pros:**
- ✅ Zero-copy view (just metadata)
- ✅ No loops needed
- ✅ Works naturally with linalg operations

**Cons:**
- ❌ **Only works for static (constant) strides**
- ❌ Most BLAS operations have dynamic strides (function parameters)

#### **Solution 3: Hybrid Approach (Recommended)**

```cpp
// In LinalgDebufferize.cpp

if (auto submapOp = dyn_cast<polygeist::SubmapOp>(user)) {
  // Extract stride operand and affine map
  Value stride = submapOp.getStride();
  AffineMap map = submapOp.getMap();
  
  // Check if stride is compile-time constant
  if (auto constStride = getConstantIntValue(stride)) {
    // Solution 2: Use tensor.extract_slice
    createExtractSliceView(submapOp, *constStride);
  } else {
    // Solution 1: Create scf.for gather loop
    createGatherLoop(submapOp, stride);
  }
}
```

### Implementation Strategy

1. **Extend `handleMemref` in LinalgDebufferize.cpp:**
   - Add case for `polygeist::SubmapOp` users
   - Transform submap into tensor operations (scf.for or extract_slice)

2. **Recursively process submap results:**
   - After materializing submap as tensor, recursively call `handleMemref` on the result
   - This allows `linalg.generic` consumers to be debufferized

3. **Pattern matching for affine maps:**
   - Simple stride: `(d0)[s0] -> (d0 * s0)` → use Solution 1 or 2
   - Offset + stride: `(d0)[s0, s1] -> (s0 + d0 * s1)` → add offset to gather loop
   - Multi-dimensional: Handle via nested loops

4. **Fusion and optimization:**
   - Let standard passes handle fusion:
     - `--scf-loop-fusion` fuses gather/compute/scatter loops
     - `--one-shot-bufferize` eliminates tensor↔memref conversions
   - Final result should be equivalent to original affine code

### Performance Considerations

**Concern:** Won't materialization introduce expensive copies?

**Answer:** No, because of fusion!

**Transformation Pipeline:**
```mlir
Step 1: Materialize (3 loops)
  %x_view = scf.for ... { gather x[i*stride] }
  %result = linalg.generic ins(%x_view) ...
  scf.for ... { scatter result[i] to y[i*stride] }

Step 2: After SCF Loop Fusion (1 loop)
  scf.for %i = 0 to %n {
    %x_val = tensor.extract %x[%i * %stride]
    %result = compute(%x_val, ...)
    tensor.insert %result into %y[%i * %stride]
  }

Step 3: After Bufferization (optimal)
  scf.for %i = 0 to %n {
    %x_val = memref.load %x[%i * %stride]
    %result = compute(%x_val, ...)
    memref.store %result, %y[%i * %stride]
  }
```

This is **exactly** the original loop - no overhead!

### Expected Behavior After Fix

```mlir
// Input: saxpy_linalg.mlir with polygeist.submap

// After linalg-debufferize:
func.func @saxpy(...) {
  %x_tensor = bufferization.to_tensor %x : memref<?xf32>
  %y_tensor = bufferization.to_tensor %y : memref<?xf32>
  
  // Materialized gather loops (to be fused)
  %x_view = scf.for %i ... { tensor.extract %x_tensor[%i * %incx] }
  %y_view = scf.for %i ... { tensor.extract %y_tensor[%i * %incy] }
  
  // Debufferized linalg.generic
  %result = linalg.generic ins(%x_view, %y_view : tensor<?xf32>, tensor<?xf32>) ...
  
  // Scatter result back
  %y_updated = scf.for %i ... { tensor.insert ... into %y_tensor[%i * %incy] }
  
  memref.copy ...
}

// After --scf-loop-fusion + --one-shot-bufferize:
func.func @saxpy(...) {
  scf.for %i = 0 to %n {
    %x_val = memref.load %x[%i * %incx]
    %y_val = memref.load %y[%i * %incy]
    %result = arith.addf %y_val, arith.mulf(%alpha, %x_val)
    memref.store %result, %y[%i * %incy]
  }
}
// Equivalent to original affine code!
```

### Impact

**Affected Operations:**
- All Level 1 BLAS operations using strided access after RaiseToLinalg
- Any `linalg.generic` operations consuming `polygeist.submap` results
- Essentially blocks the tensor-based optimization pipeline for strided operations

**Severity:** High - Prevents debufferization of common BLAS patterns, blocking tensor-level optimizations

### Related Files

- `lib/polygeist/Passes/LinalgDebufferize.cpp` - Pass implementation (needs extension)
- `saxpy_linalg.mlir` - Test case demonstrating the issue
- `saxpy_debufferized_example.mlir` - Detailed example of proposed transformation
- `blas/daxpy.c`, `blas/saxpy.c` - Original source files with strided access

### References

- MLIR Linalg Dialect: https://mlir.llvm.org/docs/Dialects/Linalg/
  - Section on indexing maps: "Maps must have no symbols"
- MLIR SCF Dialect: https://mlir.llvm.org/docs/Dialects/SCFDialect/
- Debug logs: See `temp` file, lines showing "Unknown user type (skipping): polygeist.submap"

---

## Issue #4: Dynamic Loop Offsets Not Properly Handled in SubmapOp Creation

**Date Identified:** November 11, 2025

**Status:** 🔴 Open

### Description

The `remap_in_affine_dim` function in `RaiseToLinalg.cpp` incorrectly handles dynamic (non-constant) loop lower bounds when creating `polygeist.submap` operations. When a loop has a dynamic offset (e.g., `for i = offset to offset+n`), the code attempts to extract a constant value and defaults to `0` when it fails, resulting in incorrect indexing in the generated submap.

### Error Pattern

No compilation error occurs, but the generated IR is semantically incorrect. The offset is silently ignored, leading to wrong memory accesses at runtime.

### Root Cause

In `RaiseToLinalg.cpp`, line 228 of the `remap_in_affine_dim` function:

```cpp
int lower_bound_val = getConstantFromAffineApply(lower_bound).value_or(0);
```

This code:
1. Tries to extract a **constant** value from the lower bound
2. If the lower bound is dynamic (depends on runtime values), `getConstantFromAffineApply` returns `nullopt`
3. Falls back to `0` as the default value
4. Uses this (incorrect) constant `0` in the affine map at lines 297 and 309:

```cpp
// Line 297
dimReplacements.push_back(builder.getAffineDimExpr(validDims) + 
                         builder.getAffineConstantExpr(lower_bound_val));
// Line 309
symReplacements.push_back(builder.getAffineDimExpr(validDims) + 
                         builder.getAffineConstantExpr(lower_bound_val));
```

### Example Test Case

Consider a loop with dynamic offset:

```c
void dynamic_slice(double* A, int offset, int n) {
    for (int i = offset; i < offset + n; i++) {
        A[i] = A[i] * 2.0;
    }
}
```

This generates MLIR like:

```mlir
func.func @dynamic_slice(%A: memref<?xf64>, %offset: index, %n: index) {
  %lb = affine.apply affine_map<()[s0] -> (s0)>(%offset)
  %ub = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>(%offset, %n)
  
  affine.for %i = %lb to %ub {
    %val = affine.load %A[%i] : memref<?xf64>
    %result = arith.mulf %val, %c2 : f64
    affine.store %result, %A[%i] : memref<?xf64>
  }
}
```

### Current (Incorrect) Behavior

After running `--raise-affine-to-linalg`, the code generates:

```mlir
// WRONG: offset is ignored, defaults to 0
%view = polygeist.submap(%A, %n) 
        {map = affine_map<(d0) -> (d0 + 0)>}  // Should be (d0 + offset)!
        : (memref<?xf64>, index) -> memref<?xf64>

linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>],
  iterator_types = ["parallel"]
} ins(%view : memref<?xf64>) outs(%view : memref<?xf64>) {
  // ... body ...
}
```

**Problem:** The submap starts at index `0` instead of `offset`, causing:
- Wrong memory region to be accessed
- Potential out-of-bounds access
- Incorrect computation results

### Expected Behavior

The pass should extract the dynamic offset value and pass it as a symbol operand:

```mlir
// CORRECT: offset passed as symbol operand
%view = polygeist.submap(%A, %offset, %n) 
        {map = affine_map<(d0)[s0] -> (d0 + s0)>}  // s0 = offset
        : (memref<?xf64>, index, index) -> memref<?xf64>

linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>],
  iterator_types = ["parallel"]
} ins(%view : memref<?xf64>) outs(%view : memref<?xf64>) {
  // ... body ...
}
```

Where:
- `%offset` is passed as a symbol operand to the submap
- The map `affine_map<(d0)[s0] -> (d0 + s0)>` correctly adds the offset
- When indexing `view[i]`, it computes `A[i + offset]` as intended

### Why This Design Works

The current `SubmapOp` design is actually **correct** and doesn't need explicit offset operands because:

1. **Constant offsets:** Can be baked into the affine map as constant expressions
   ```mlir
   map = affine_map<(d0) -> (d0 + 10)>  // Constant offset 10
   ```

2. **Dynamic offsets:** Can be passed as symbol operands and referenced in the map
   ```mlir
   polygeist.submap(%base, %offset, %size) 
   map = affine_map<(d0)[s0] -> (d0 + s0)>  // s0 is the dynamic offset
   ```

The bug is NOT a design limitation - it's an implementation bug in handling dynamic offsets.

### Proposed Fix

Modify `remap_in_affine_dim` in `RaiseToLinalg.cpp`:

```cpp
// Current (line 228):
int lower_bound_val = getConstantFromAffineApply(lower_bound).value_or(0);

// Proposed fix:
std::optional<int64_t> lower_bound_const = getConstantFromAffineApply(lower_bound);
if (lower_bound_const) {
  // Constant offset: bake into the map
  int lower_bound_val = *lower_bound_const;
  dimReplacements.push_back(builder.getAffineDimExpr(validDims) + 
                           builder.getAffineConstantExpr(lower_bound_val));
  validDims++;
} else {
  // Dynamic offset: add as symbol operand
  Value lowerBoundValue = lower_bound.getResult();
  
  // Add the offset to operands_without_indices
  operands_without_indices.push_back(lowerBoundValue);
  
  // Reference it as a symbol in the map
  dimReplacements.push_back(builder.getAffineDimExpr(validDims) + 
                           builder.getAffineSymbolExpr(validSims));
  validDims++;
  validSims++;
}
```

Similar changes needed for the symbol replacement logic (lines 306-315).

### Alternative: More Robust Extraction

Instead of assuming `AffineApplyOp` represents the lower bound simply, extract the actual lower bound operands and map:

```cpp
// Extract the lower bound map and its operands
AffineMap lbMap = lower_bound.getAffineMap();
ValueRange lbOperands = lower_bound.getOperands();

// Check if it's a constant map
if (lbMap.getNumResults() == 1 && lbMap.isSingleConstant()) {
  // Constant offset: bake into map
  int64_t offset = lbMap.getSingleConstantResult();
  // ... add as constant expression ...
} else {
  // Dynamic offset: add operands as symbols
  for (Value operand : lbOperands) {
    operands_without_indices.push_back(operand);
  }
  // ... update map to reference these symbols ...
}
```

### Impact

**Affected Operations:**
- Any affine loops with dynamic lower bounds (not starting at constant 0)
- Loops iterating over sub-slices of arrays (e.g., `for i = start to end`)
- Tiled loop nests where tile offsets are runtime parameters
- Blocked algorithms with dynamic block boundaries

**Severity:** High - Silent correctness bug that produces wrong results without any error message

**Current Workaround:** 
- Manually rewrite loops to always start at 0 and adjust indexing
- Example: `for (i = offset; i < offset+n; i++) A[i] = ...` 
  → `for (i = 0; i < n; i++) A[i+offset] = ...`
- This workaround may not be feasible for all patterns

### Related Files

- `lib/polygeist/Passes/RaiseToLinalg.cpp` - Lines 218-435 (`remap_in_affine_dim` function)
  - Line 228: Incorrect default to `0` for dynamic offsets
  - Lines 297, 309: Where the (incorrect) constant offset is used
- `include/polygeist/PolygeistOps.td` - Lines 318-369 (`SubmapOp` definition)
  - Shows that the design supports symbol operands for dynamic values

### Testing Strategy

Create test cases with:
1. **Simple dynamic offset:**
   ```c
   for (int i = offset; i < offset + n; i++) A[i] = B[i];
   ```

2. **Nested loops with dynamic offsets:**
   ```c
   for (int i = i_start; i < i_end; i++)
     for (int j = j_start; j < j_end; j++)
       C[i][j] = A[i][j] + B[i][j];
   ```

3. **Tiled GEMM with dynamic tile offsets:**
   ```c
   for (int ii = tile_i; ii < tile_i + tile_size; ii++)
     for (int jj = tile_j; jj < tile_j + tile_size; jj++)
       // ... computation ...
   ```

Verify that the generated `polygeist.submap` operations:
- Include offset values as symbol operands
- Have affine maps that correctly reference these symbols
- Produce correct results when lowered and executed

### References

- MLIR Affine Dialect Documentation: https://mlir.llvm.org/docs/Dialects/Affine/
- Affine Map composition and symbol handling
- Related to Issue #1 (strided access) - both involve properly encoding indexing information in submaps

---

## Issue #5: SubmapInverse Cannot Handle Shape-Changing Tensor Transformations

**Date Identified:** November 11, 2025

**Status:** 🔴 Open

### Description

The `polygeist.submapInverse` operation cannot correctly scatter results back when the tensor undergoes shape-changing transformations (like `tensor.expand_shape`, `tensor.collapse_shape`, or `tensor.extract_slice`) between the initial `polygeist.submap` and the final `submapInverse`. This limitation affects the debufferization pipeline when intermediate optimization passes modify tensor shapes.

### Error Pattern

Type mismatch errors occur when `submapInverse` receives a tensor of different shape than expected:

```
error: 'polygeist.submapInverse' op operand #1 must be memref of any type values or tensor of any type values, but got 'tensor<5x5xf64>'
```

Or semantic errors where the scatter-back operation cannot determine the correct mapping from the transformed shape back to the original strided view.

### Root Cause

The `submapInverse` operation expects to receive a tensor with the **exact same shape** as the original `submap` result. When intermediate operations change the shape, the type signature no longer matches:

```mlir
// Original submap creates tensor<50xf64>
%view = polygeist.submap(%base_tensor, %stride, %c50) 
        {map = affine_map<(d0)[s0] -> (d0 * s0)>}
        : (tensor<100xf64>, index, index) -> tensor<50xf64>

// Shape change: tensor<50xf64> → tensor<10x5xf64>
%view_2d = tensor.expand_shape %view [[0, 1]] 
           : tensor<50xf64> into tensor<10x5xf64>

// Computation on transformed shape
%result_2d = linalg.generic ... outs(%view_2d : tensor<10x5xf64>) -> tensor<10x5xf64>

// ❌ TYPE MISMATCH: submapInverse expects tensor<50xf64>, receives tensor<10x5xf64>
%updated = polygeist.submapInverse(%base_tensor, %result_2d, %stride, %c50)
           {map = affine_map<(d0)[s0] -> (d0 * s0)>}
           : (tensor<100xf64>, tensor<10x5xf64>, index, index) -> tensor<100xf64>
```

### When Does This Work?

`submapInverse` works correctly when the tensor chain **preserves shape**:

✅ **Safe Operations (Shape-Preserving):**
- `linalg.generic` with identity indexing maps
- `linalg.map` (element-wise operations)
- Element-wise arithmetic (`arith.addf`, `arith.mulf`, etc.)
- `tensor.insert` / `tensor.extract` (single elements)
- `math.*` operations (element-wise)

### When Does This Break?

❌ **Problematic Operations (Shape-Changing):**
- `tensor.expand_shape` / `tensor.collapse_shape` - Changes dimensionality
- `tensor.extract_slice` - Reduces dimensions or size
- `tensor.insert_slice` - If inserting into different shape
- `tensor.pad` - Changes tensor size
- `tensor.concat` - Changes tensor size
- `tensor.broadcast` - Adds dimensions

### Example Test Case

**Memref Version (Works Fine):**
```mlir
func.func @reshape_memref(%A: memref<100xf64>, %stride: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c50 = arith.constant 50 : index
  %c2 = arith.constant 2.0 : f64
  
  scf.for %i = %c0 to %c50 step %c1 {
    %idx = arith.muli %i, %stride : index
    %val = memref.load %A[%idx] : memref<100xf64>
    %doubled = arith.mulf %val, %c2 : f64
    memref.store %doubled, %A[%idx] : memref<100xf64>
  }
  
  return
}
```

**Tensor Version with Reshape (Breaks):**
```mlir
func.func @reshape_tensor(%A: memref<100xf64>, %stride: index) {
  %c50 = arith.constant 50 : index
  %c2 = arith.constant 2.0 : f64
  
  %A_tensor = bufferization.to_tensor %A : memref<100xf64>
  
  %A_view = polygeist.submap(%A_tensor, %stride, %c50) 
            {map = affine_map<(d0)[s0] -> (d0 * s0)>}
            : (tensor<100xf64>, index, index) -> tensor<50xf64>
  
  %A_2d = tensor.expand_shape %A_view [[0, 1]] 
          : tensor<50xf64> into tensor<10x5xf64>
  
  %result_2d = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%A_2d : tensor<10x5xf64>) {
  ^bb0(%val: f64):
    %doubled = arith.mulf %val, %c2 : f64
    linalg.yield %doubled : f64
  } -> tensor<10x5xf64>
  
  %A_updated = polygeist.submapInverse(%A_tensor, %result_2d, %stride, %c50)
               {map = affine_map<(d0)[s0] -> (d0 * s0)>}
               : (tensor<100xf64>, tensor<10x5xf64>, index, index) -> tensor<100xf64>
  
  %A_memref = bufferization.to_memref %A_updated : memref<100xf64>
  memref.copy %A_memref, %A : memref<100xf64> to memref<100xf64>
  
  return
}
```

**Problem:** The `tensor.expand_shape` changes `tensor<50xf64>` to `tensor<10x5xf64>`, but `submapInverse` still expects `tensor<50xf64>`.

### Why Does This Happen?

Tensor-level optimization passes may introduce shape changes for various reasons:
- **Vectorization preparation**: Reshaping to expose parallelism
- **Tiling**: Splitting dimensions for cache locality
- **Layout optimization**: Changing data layout for better memory access
- **Broadcasting**: Expanding dimensions for element-wise operations

These transformations are valid and beneficial in tensor land, but they break the assumption that `submapInverse` can directly map the result back to the original strided view.

### Proposed Solutions

#### **Solution 1: Restrict Debufferization to Shape-Preserving Chains**

Only debufferize when the tensor chain between `submap` and `submapInverse` preserves shape:

```cpp
// In LinalgDebufferize.cpp
bool isShapePreserving(Operation* op) {
  return isa<linalg::GenericOp, linalg::MapOp, arith::ArithOp, math::MathOp>(op);
}

// When processing submap:
if (auto submapOp = dyn_cast<polygeist::SubmapOp>(user)) {
  // Check if all uses preserve shape
  for (Operation* user : submapOp->getUsers()) {
    if (!isShapePreserving(user)) {
      return failure(); // Skip debufferization
    }
  }
  // Proceed with debufferization...
}
```

**Pros:**
- ✅ Simple to implement
- ✅ Guarantees correctness
- ✅ Works for common BLAS patterns (element-wise operations)

**Cons:**
- ❌ Misses optimization opportunities when shape changes are beneficial
- ❌ Requires conservative analysis

#### **Solution 2: Track Provenance and Insert Inverse Transformations**

Track the chain of transformations and insert inverse operations before `submapInverse`:

```mlir
// Original chain:
%view = polygeist.submap(...) -> tensor<50xf64>
%view_2d = tensor.expand_shape %view [[0, 1]] -> tensor<10x5xf64>
%result_2d = linalg.generic ... -> tensor<10x5xf64>

// Insert inverse transformation:
%result_1d = tensor.collapse_shape %result_2d [[0, 1]] -> tensor<50xf64>
%updated = polygeist.submapInverse(%base, %result_1d, ...) -> tensor<100xf64>
```

**Pros:**
- ✅ Allows shape-changing optimizations
- ✅ Maintains correctness by undoing transformations

**Cons:**
- ❌ Complex provenance tracking required
- ❌ May not always be invertible (e.g., `extract_slice` loses information)
- ❌ Additional overhead from inverse operations

#### **Solution 3: Use SubmapInverse at Original Tensor Shape**

Instead of threading the transformed tensor through, maintain a parallel chain for the scatter-back:

```mlir
// Keep original view for scatter-back
%view = polygeist.submap(%base, %stride, %size) -> tensor<50xf64>

// Create transformed version for computation
%view_2d = tensor.expand_shape %view [[0, 1]] -> tensor<10x5xf64>
%result_2d = linalg.generic ... outs(%view_2d) -> tensor<10x5xf64>

// Collapse back before scatter
%result_1d = tensor.collapse_shape %result_2d [[0, 1]] -> tensor<50xf64>

// Scatter with matching shape
%updated = polygeist.submapInverse(%base, %result_1d, %stride, %size)
```

This is similar to Solution 2 but explicitly managed by the debufferization pass.

#### **Solution 4: Extend SubmapInverse to Handle Shape Metadata**

Extend `submapInverse` to accept shape transformation metadata:

```mlir
%updated = polygeist.submapInverse(%base, %result_2d, %stride, %size)
           {map = affine_map<(d0)[s0] -> (d0 * s0)>,
            shape_transform = #polygeist.reshape<[50] -> [10, 5]>}
           : (tensor<100xf64>, tensor<10x5xf64>, index, index) -> tensor<100xf64>
```

**Pros:**
- ✅ Most general solution
- ✅ Encapsulates complexity in the operation

**Cons:**
- ❌ Significant implementation complexity
- ❌ Requires extending the operation definition
- ❌ May be overkill for common cases

### Recommended Approach

**Start with Solution 1 (Restricted Debufferization):**
1. Implement debufferization only for shape-preserving chains
2. Add validation to detect and reject shape-changing operations
3. Document the limitation clearly

**Future Enhancement (Solution 2/3):**
1. Add provenance tracking to detect shape transformations
2. Insert inverse transformations before `submapInverse`
3. Validate that transformations are invertible

### Implementation Strategy

```cpp
// In LinalgDebufferize.cpp

// Helper to check if operation preserves shape
static bool preservesShape(Operation* op) {
  return isa<linalg::GenericOp, linalg::MapOp>(op) ||
         op->hasTrait<OpTrait::Elementwise>() ||
         isa<arith::ArithOp, math::MathOp>(op);
}

// When handling submap:
if (auto submapOp = dyn_cast<polygeist::SubmapOp>(user)) {
  // Validate shape preservation
  SmallVector<Operation*> chain;
  if (!collectShapePreservingChain(submapOp, chain)) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping submap: chain contains shape-changing ops\n");
    return failure();
  }
  
  // Proceed with debufferization...
  Value tensorView = convertSubmapToTensor(submapOp);
  // ... process chain ...
  insertSubmapInverse(submapOp, tensorView);
}
```

### Impact

**Affected Operations:**
- Any debufferization pipeline that includes shape-changing tensor operations
- Optimization passes that reshape tensors for vectorization or tiling
- Patterns that use `tensor.expand_shape`, `tensor.collapse_shape`, or `tensor.extract_slice`

**Severity:** Medium - Limits the applicability of debufferization but doesn't cause correctness issues (operations are rejected rather than producing wrong results)

**Current Workaround:**
- Avoid shape-changing operations in the tensor chain between `submap` and `submapInverse`
- Perform shape transformations before or after the debufferized region
- Use memref-based transformations instead of tensor-based ones

### Related Files

- `lib/polygeist/Passes/LinalgDebufferize.cpp` - Needs validation logic
- `include/polygeist/PolygeistOps.td` - `SubmapOp` and `SubmapInverseOp` definitions
- `saxpy_debufferized_example.mlir` - Example showing shape-preserving debufferization

### Related Issues

- **Issue #3**: LinalgDebufferize Cannot Handle polygeist.submap Operations (parent issue)
- This issue documents a specific limitation once Issue #3 is resolved

### References

- MLIR Tensor Dialect: https://mlir.llvm.org/docs/Dialects/TensorOps/
- Shape manipulation operations and their semantics
- Provenance tracking in compiler transformations

---

## Future Issues

*Document additional issues here as they are discovered.*

