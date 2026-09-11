# Understanding polygeist.submapInverse

This document explains why `polygeist.submapInverse` is essential for the debufferization pipeline and how it works.

---

## Table of Contents

1. [The Problem: Strided Scatter-Back](#the-problem-strided-scatter-back)
2. [What submapInverse Does](#what-submapinverse-does)
3. [Complete Example: With vs Without submapInverse](#complete-example-with-vs-without-submapinverse)
4. [How submapInverse Works Internally](#how-submapinverse-works-internally)
5. [Why to_memref + copy is Still Needed](#why-to_memref--copy-is-still-needed)
6. [Multiple submapInverse Operations](#multiple-submapinverse-operations)
7. [After Bufferization: Final Optimized Code](#after-bufferization-final-optimized-code)

---

## The Problem: Strided Scatter-Back

When we debufferize operations on strided views, we face a fundamental challenge:

**The Gather-Compute-Scatter Pattern:**

```
Original memref:  [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, ...]  (100 elements)
                   ^       ^       ^       ^       ^
                   |       |       |       |       |
Stride=2 extracts: a0      a2      a4      a6      a8           (50 elements)
                   ↓       ↓       ↓       ↓       ↓
Contiguous view:  [a0, a2, a4, a6, a8, ...]                     (50 elements)
                   ↓       ↓       ↓       ↓       ↓
Compute (×2):     [v0, v1, v2, v3, v4, ...]                     (50 elements)
                   ↓       ↓       ↓       ↓       ↓
Scatter back:      v0      v1      v2      v3      v4
                   ↓       ↓       ↓       ↓       ↓
Updated memref:   [v0, a1, v1, a3, v2, a5, v3, a7, v4, a9, ...]
```

**The challenge:** How do we scatter 50 contiguous values back to 50 non-contiguous (strided) positions?

---

## What submapInverse Does

`polygeist.submapInverse` performs a **strided scatter** operation:

```mlir
%result = polygeist.submapInverse(%base, %values, %stride, %size)
          {map = affine_map<(d0)[s0] -> (d0 * s0)>}
          : (tensor<100xf64>, tensor<50xf64>, index, index) -> tensor<100xf64>
```

**Semantics:**
1. Takes a base tensor (`tensor<100xf64>`)
2. Takes computed values (`tensor<50xf64>`) - contiguous
3. Scatters values to strided positions using the affine map
4. Preserves all other elements in the base tensor
5. Returns a new tensor with updates applied

**Key properties:**
- Uses the **same affine map** as the original `polygeist.submap`
- Inverse operation: `submap` gathers, `submapInverse` scatters
- Preserves elements not covered by the strided access pattern

---

## Complete Example: With vs Without submapInverse

### Scenario: Double every other element (stride=2)

**Input:** `A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]`  
**Goal:** Double elements at positions 0, 2, 4, 6, 8 (stride=2)  
**Expected:** `A = [2.0, 2.0, 6.0, 4.0, 10.0, 6.0, 14.0, 8.0, 18.0, 10.0]`

### ❌ Without submapInverse: Manual Scatter Loop

```mlir
func.func @without_inverse(%A: memref<10xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %c2_f64 = arith.constant 2.0 : f64
  
  // Convert to tensor
  %A_tensor = bufferization.to_tensor %A : memref<10xf64>
  
  // Gather: Extract strided elements
  %A_view = polygeist.submap(%A_tensor, %c2, %c5) 
            {map = affine_map<(d0)[s0] -> (d0 * s0)>}
            : (tensor<10xf64>, index, index) -> tensor<5xf64>
  // %A_view = [1.0, 3.0, 5.0, 7.0, 9.0]
  
  // Compute: Double the values
  %computed = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } outs(%A_view : tensor<5xf64>) {
  ^bb0(%a: f64):
    %doubled = arith.mulf %a, %c2_f64 : f64
    linalg.yield %doubled : f64
  } -> tensor<5xf64>
  // %computed = [2.0, 6.0, 10.0, 14.0, 18.0]
  
  // ❌ PROBLEM: How to scatter back?
  // Can't do: memref.copy %computed, %A  (shape mismatch: 5 vs 10)
  
  // Manual scatter loop required:
  %computed_memref = bufferization.to_memref %computed : memref<5xf64>
  
  scf.for %i = %c0 to %c5 step %c1 {
    // Load from contiguous position i
    %val = memref.load %computed_memref[%i] : memref<5xf64>
    
    // Store to strided position i * 2
    %strided_idx = arith.muli %i, %c2 : index
    memref.store %val, %A[%strided_idx] : memref<10xf64>
  }
  // Manual loop: A[0]=2.0, A[2]=6.0, A[4]=10.0, A[6]=14.0, A[8]=18.0
  
  return
}
```

**Problems:**
- ❌ Verbose: Requires manual scatter loop
- ❌ Error-prone: Easy to get indexing wrong
- ❌ Loses semantic information: Compiler can't see the strided pattern
- ❌ Harder to optimize: Loop fusion, vectorization passes may miss opportunities

### ✅ With submapInverse: Clean and Declarative

```mlir
func.func @with_inverse(%A: memref<10xf64>) {
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %c2_f64 = arith.constant 2.0 : f64
  
  // Convert to tensor
  %A_tensor = bufferization.to_tensor %A : memref<10xf64>
  
  // Gather: Extract strided elements
  %A_view = polygeist.submap(%A_tensor, %c2, %c5) 
            {map = affine_map<(d0)[s0] -> (d0 * s0)>}
            : (tensor<10xf64>, index, index) -> tensor<5xf64>
  // %A_view = [1.0, 3.0, 5.0, 7.0, 9.0]
  
  // Compute: Double the values
  %computed = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } outs(%A_view : tensor<5xf64>) {
  ^bb0(%a: f64):
    %doubled = arith.mulf %a, %c2_f64 : f64
    linalg.yield %doubled : f64
  } -> tensor<5xf64>
  // %computed = [2.0, 6.0, 10.0, 14.0, 18.0]
  
  // ✅ Scatter: Use submapInverse with SAME map
  %A_updated = polygeist.submapInverse(%A_tensor, %computed, %c2, %c5)
               {map = affine_map<(d0)[s0] -> (d0 * s0)>}
               : (tensor<10xf64>, tensor<5xf64>, index, index) -> tensor<10xf64>
  // %A_updated = [2.0, 2.0, 6.0, 4.0, 10.0, 6.0, 14.0, 8.0, 18.0, 10.0]
  //               ^new  ^old ^new  ^old ^new   ^old ^new   ^old ^new    ^old
  
  // Convert back to memref and copy
  %A_final = bufferization.to_memref %A_updated : memref<10xf64>
  memref.copy %A_final, %A : memref<10xf64> to memref<10xf64>
  
  return
}
```

**Advantages:**
- ✅ Clean: Single operation for strided scatter
- ✅ Declarative: Clearly expresses the intent
- ✅ Preserves semantics: Compiler knows it's a strided scatter
- ✅ Optimizable: Subsequent passes can fuse and vectorize

---

## How submapInverse Works Internally

### Step-by-Step Execution

```mlir
%A_updated = polygeist.submapInverse(%A_tensor, %computed, %stride, %size)
             {map = affine_map<(d0)[s0] -> (d0 * s0)>}
```

**Given:**
- `%A_tensor = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]` (10 elements)
- `%computed = [v0, v1, v2, v3, v4]` (5 elements)
- `%stride = 2`
- `%size = 5`
- `map = affine_map<(d0)[s0] -> (d0 * s0)>`

**Process:**

```
For each dimension index d0 in [0, size):
  1. Compute target index: target = map(d0)[stride] = d0 * stride
  2. Update: result[target] = computed[d0]
  3. Preserve: result[i] = A_tensor[i] for all i != target

Iteration 0: d0=0 → target=0*2=0 → result[0] = v0
Iteration 1: d0=1 → target=1*2=2 → result[2] = v1
Iteration 2: d0=2 → target=2*2=4 → result[4] = v2
Iteration 3: d0=3 → target=3*2=6 → result[6] = v3
Iteration 4: d0=4 → target=4*2=8 → result[8] = v4

Preserve: result[1]=a1, result[3]=a3, result[5]=a5, result[7]=a7, result[9]=a9
```

**Result:**
```
%A_updated = [v0, a1, v1, a3, v2, a5, v3, a7, v4, a9]
```

### With Offset + Stride

```mlir
%result = polygeist.submapInverse(%base, %values, %offset, %stride, %size)
          {map = affine_map<(d0)[s0, s1] -> (s0 + d0 * s1)>}
          : (tensor<100xf64>, tensor<30xf64>, index, index, index) -> tensor<100xf64>
```

**Given:**
- `%base = [b0, b1, b2, ..., b99]` (100 elements)
- `%values = [v0, v1, v2, ..., v29]` (30 elements)
- `%offset = 10`
- `%stride = 3`
- `map = affine_map<(d0)[s0, s1] -> (s0 + d0 * s1)>`

**Process:**

```
For d0 in [0, 30):
  target = offset + d0 * stride = 10 + d0 * 3

d0=0 → target=10+0*3=10 → result[10] = v0
d0=1 → target=10+1*3=13 → result[13] = v1
d0=2 → target=10+2*3=16 → result[16] = v2
...
d0=29 → target=10+29*3=97 → result[97] = v29

Preserve: All other elements remain unchanged
```

---

## Why to_memref + copy is Still Needed

Even with `submapInverse`, we still need the final `to_memref` + `copy`. Here's why:

### The Type System Boundary

```mlir
func.func @example(%A: memref<100xf64>, %stride: index) {
  //                 ^^^^^^^^^^^^^^^^ Input is MEMREF
  
  // Enter tensor world
  %A_tensor = bufferization.to_tensor %A : memref<100xf64>
  
  // ... submap, compute, submapInverse ...
  
  %A_updated = polygeist.submapInverse(...)
               : (...) -> tensor<100xf64>
  //                      ^^^^^^^^^^^^^^ Returns TENSOR
  
  // Exit tensor world
  %A_final = bufferization.to_memref %A_updated : memref<100xf64>
  
  // Update the original memref parameter
  memref.copy %A_final, %A : memref<100xf64> to memref<100xf64>
  //          ^^^^^^^^  ^^
  //          new data  original parameter
  
  return
}
```

### Memref vs Tensor Semantics

**Memrefs (Reference Semantics):**
- Like pointers or references
- Operations modify data in-place
- Multiple memrefs can alias the same memory
- Mutable

**Tensors (Value Semantics):**
- Like immutable values
- Operations create new tensors
- No aliasing
- Immutable

### What Happens Without the Copy

```mlir
func.func @no_copy(%A: memref<100xf64>, %stride: index) {
  %A_tensor = bufferization.to_tensor %A : memref<100xf64>
  
  %A_view = polygeist.submap(%A_tensor, %stride, %c50) ...
  %computed = linalg.generic ... -> tensor<50xf64>
  %A_updated = polygeist.submapInverse(%A_tensor, %computed, ...) 
               -> tensor<100xf64>
  
  %A_final = bufferization.to_memref %A_updated : memref<100xf64>
  
  // ❌ NO COPY - Original %A is never updated!
  return
}

// Caller sees:
func.func @caller() {
  %A = memref.alloc() : memref<100xf64>
  // ... initialize A with [1, 2, 3, ...] ...
  
  call @no_copy(%A, %stride) : (memref<100xf64>, index) -> ()
  
  // ❌ %A still has [1, 2, 3, ...] - unchanged!
  // The computation was lost because we never copied back
}
```

### The Complete Flow

```
Input memref %A
    ↓
to_tensor (enter tensor world)
    ↓
submap (gather strided elements)
    ↓
linalg.generic (compute)
    ↓
submapInverse (scatter back to tensor)
    ↓
to_memref (exit tensor world, creates new memref)
    ↓
memref.copy (update original memref parameter)
    ↓
Original %A is now updated
```

**Each step is necessary:**
1. `to_tensor`: Enter tensor world for transformations
2. `submap`: Gather strided elements into contiguous tensor
3. `linalg.generic`: Perform computation
4. `submapInverse`: Scatter results back to strided positions
5. `to_memref`: Exit tensor world
6. `memref.copy`: Update the original input parameter

---

## Multiple submapInverse Operations

When multiple arrays are modified, each needs its own `submapInverse`:

```mlir
func.func @multi_array(%X: memref<200xf64>, %Y: memref<300xf64>,
                       %stride_x: index, %stride_y: index) {
  %c50 = arith.constant 50 : index
  %c2 = arith.constant 2.0 : f64
  
  %X_tensor = bufferization.to_tensor %X : memref<200xf64>
  %Y_tensor = bufferization.to_tensor %Y : memref<300xf64>
  
  // Create two submaps with different strides
  %X_view = polygeist.submap(%X_tensor, %stride_x, %c50) 
            {map = affine_map<(d0)[s0] -> (d0 * s0)>}
            : (tensor<200xf64>, index, index) -> tensor<50xf64>
  
  %Y_view = polygeist.submap(%Y_tensor, %stride_y, %c50) 
            {map = affine_map<(d0)[s0] -> (d0 * s0)>}
            : (tensor<300xf64>, index, index) -> tensor<50xf64>
  
  // Compute: Y = Y + X * 2
  %Y_result = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,  // X_view
      affine_map<(d0) -> (d0)>   // Y_view
    ],
    iterator_types = ["parallel"]
  } ins(%X_view : tensor<50xf64>) outs(%Y_view : tensor<50xf64>) {
  ^bb0(%x: f64, %y: f64):
    %scaled = arith.mulf %x, %c2 : f64
    %sum = arith.addf %y, %scaled : f64
    linalg.yield %sum : f64
  } -> tensor<50xf64>
  
  // Compute: X = X * 3
  %X_result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } outs(%X_view : tensor<50xf64>) {
  ^bb0(%x: f64):
    %c3 = arith.constant 3.0 : f64
    %tripled = arith.mulf %x, %c3 : f64
    linalg.yield %tripled : f64
  } -> tensor<50xf64>
  
  // Scatter Y back using stride_y
  %Y_updated = polygeist.submapInverse(%Y_tensor, %Y_result, %stride_y, %c50)
               {map = affine_map<(d0)[s0] -> (d0 * s0)>}
               : (tensor<300xf64>, tensor<50xf64>, index, index) -> tensor<300xf64>
  
  // Scatter X back using stride_x
  %X_updated = polygeist.submapInverse(%X_tensor, %X_result, %stride_x, %c50)
               {map = affine_map<(d0)[s0] -> (d0 * s0)>}
               : (tensor<200xf64>, tensor<50xf64>, index, index) -> tensor<200xf64>
  
  // Convert and copy both
  %Y_final = bufferization.to_memref %Y_updated : memref<300xf64>
  %X_final = bufferization.to_memref %X_updated : memref<200xf64>
  
  memref.copy %Y_final, %Y : memref<300xf64> to memref<300xf64>
  memref.copy %X_final, %X : memref<200xf64> to memref<200xf64>
  
  return
}
```

**Key points:**
- Each modified array needs its own `submapInverse`
- Each uses its own stride and map
- Read-only arrays (inputs only) don't need `submapInverse`

---

## After Bufferization: Final Optimized Code

The good news: After running **one-shot bufferization**, all the tensor operations and conversions get optimized away!

### Before One-Shot Bufferization

```mlir
func.func @example(%A: memref<100xf64>, %stride: index) {
  %c50 = arith.constant 50 : index
  %c2 = arith.constant 2.0 : f64
  
  %A_tensor = bufferization.to_tensor %A : memref<100xf64>
  
  %A_view = polygeist.submap(%A_tensor, %stride, %c50) 
            {map = affine_map<(d0)[s0] -> (d0 * s0)>}
            : (tensor<100xf64>, index, index) -> tensor<50xf64>
  
  %computed = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } outs(%A_view : tensor<50xf64>) {
  ^bb0(%a: f64):
    %doubled = arith.mulf %a, %c2 : f64
    linalg.yield %doubled : f64
  } -> tensor<50xf64>
  
  %A_updated = polygeist.submapInverse(%A_tensor, %computed, %stride, %c50)
               {map = affine_map<(d0)[s0] -> (d0 * s0)>}
               : (tensor<100xf64>, tensor<50xf64>, index, index) -> tensor<100xf64>
  
  %A_final = bufferization.to_memref %A_updated : memref<100xf64>
  memref.copy %A_final, %A : memref<100xf64> to memref<100xf64>
  
  return
}
```

### After One-Shot Bufferization

```mlir
func.func @example(%A: memref<100xf64>, %stride: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c50 = arith.constant 50 : index
  %c2 = arith.constant 2.0 : f64
  
  // All tensor ops are gone! Direct in-place memref operations
  scf.for %i = %c0 to %c50 step %c1 {
    // Compute strided index
    %idx = arith.muli %i, %stride : index
    
    // Load from strided position
    %val = memref.load %A[%idx] : memref<100xf64>
    
    // Compute
    %doubled = arith.mulf %val, %c2 : f64
    
    // Store back to strided position (in-place)
    memref.store %doubled, %A[%idx] : memref<100xf64>
  }
  
  return
}
```

**What happened:**
- ✅ All tensor operations eliminated
- ✅ Direct in-place memref operations
- ✅ No copies, no allocations
- ✅ Optimal performance - equivalent to hand-written code

**Further optimization (after loop fusion, vectorization):**

```mlir
func.func @example(%A: memref<100xf64>, %stride: index) {
  %c0 = arith.constant 0 : index
  %c50 = arith.constant 50 : index
  %c2_vec = arith.constant dense<2.0> : vector<4xf64>
  
  // Vectorized strided load-compute-store
  scf.for %i = %c0 to %c50 step %c4 {
    %vec = vector.strided_load %A[%i], %stride : vector<4xf64>
    %doubled = arith.mulf %vec, %c2_vec : vector<4xf64>
    vector.strided_store %doubled, %A[%i], %stride : vector<4xf64>
  }
  
  return
}
```

---

## Summary

### Why submapInverse is Essential

1. **Solves the scatter problem**: Efficiently scatters contiguous values to strided positions
2. **Preserves semantics**: Maintains high-level information about strided access patterns
3. **Enables optimization**: Allows subsequent passes to recognize and optimize strided operations
4. **Clean abstraction**: Avoids manual scatter loops, reducing verbosity and errors

### The Complete Transformation Pipeline

```
Original memref code (strided in-place operations)
    ↓
LinalgDebufferize (add submap/submapInverse)
    ↓
Tensor-based IR (functional, immutable)
    ↓
Tensor optimizations (fusion, tiling, etc.)
    ↓
One-shot bufferization
    ↓
Optimized memref code (back to in-place operations)
    ↓
Vectorization, lowering
    ↓
Efficient machine code
```

### Key Takeaways

- **`submap`**: Gathers strided elements into contiguous tensor
- **`submapInverse`**: Scatters contiguous tensor back to strided positions
- **`to_memref` + `copy`**: Bridges tensor world back to memref world
- **After bufferization**: All overhead disappears, leaving optimal code

The debufferization → optimization → rebufferization cycle allows us to apply powerful tensor-level transformations to strided memref operations while maintaining correctness and achieving optimal performance!

