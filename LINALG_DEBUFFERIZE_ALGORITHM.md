# LinalgDebufferize Pass - Algorithm Documentation

**File:** `lib/polygeist/Passes/LinalgDebufferize.cpp`  
**Author:** Polygeist Team  
**Last Updated:** October 17, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Purpose and Goals](#purpose-and-goals)
3. [High-Level Transformation](#high-level-transformation)
4. [Algorithm Flowchart](#algorithm-flowchart)
5. [Detailed Algorithm](#detailed-algorithm)
6. [Key Data Structures](#key-data-structures)
7. [Helper Functions](#helper-functions)
8. [Region Propagation](#region-propagation)
9. [Example Transformations](#example-transformations)
10. [Current Limitations](#current-limitations)
11. [Future Work](#future-work)

---

## Overview

The `LinalgDebufferize` pass transforms **memref-based** Linalg operations into **tensor-based** operations. This enables:

- **Fusion opportunities**: Tensor operations can be fused more easily
- **High-level optimizations**: Polyhedral transformations, tiling, vectorization
- **Explicit data flow**: SSA form makes dependencies clear
- **Functional semantics**: Easier to reason about and optimize

### Pattern Structure

The pass uses **greedy pattern rewriting** with two main patterns:

1. **`LinalgDebufferization`**: Main transformation pattern (matches `func::FuncOp`)
2. **`debufferizationAllocaRemoval`**: Cleanup pattern for dead allocations

---

## Purpose and Goals

### Input: Memref-based Linalg
```mlir
func.func @example(%A: memref<10xf32>, %B: memref<10xf32>) {
  %alloca = memref.alloca() : memref<10xf32>
  linalg.generic ins(%A : memref<10xf32>) 
                 outs(%alloca : memref<10xf32>) {
    ^bb0(%a: f32, %out: f32):
      %c = arith.mulf %a, %a : f32
      linalg.yield %c : f32
  }
  linalg.generic ins(%alloca : memref<10xf32>) 
                 outs(%B : memref<10xf32>) {
    ^bb0(%a: f32, %out: f32):
      %c = arith.addf %a, %a : f32
      linalg.yield %c : f32
  }
  return
}
```

### Output: Tensor-based Linalg
```mlir
func.func @example(%A: memref<10xf32>, %B: memref<10xf32>) {
  %A_tensor = bufferization.to_tensor %A : memref<10xf32>
  
  %init1 = tensor.empty() : tensor<10xf32>
  %result1 = linalg.generic ins(%A_tensor : tensor<10xf32>) 
                            outs(%init1 : tensor<10xf32>) -> tensor<10xf32> {
    ^bb0(%a: f32, %out: f32):
      %c = arith.mulf %a, %a : f32
      linalg.yield %c : f32
  }
  
  %init2 = tensor.empty() : tensor<10xf32>
  %result2 = linalg.generic ins(%result1 : tensor<10xf32>) 
                            outs(%init2 : tensor<10xf32>) -> tensor<10xf32> {
    ^bb0(%a: f32, %out: f32):
      %c = arith.addf %a, %a : f32
      linalg.yield %c : f32
  }
  
  %B_memref = bufferization.to_memref %result2 : memref<10xf32>
  memref.copy %B_memref, %B : memref<10xf32> to memref<10xf32>
  return
}
```

**Key Differences:**
- Memref operations are **in-place** (side effects)
- Tensor operations are **functional** (return new values)
- Tensor form enables **fusion** (subsequent passes can merge the two generics)

---

## High-Level Transformation

```
┌─────────────────────────────────────────────────────────────────┐
│                    LinalgDebufferize Pass                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Identify Memref Roots to Debufferize  │
        │  • memref.alloca operations             │
        │  • memref.alloc operations              │
        │  • Function arguments (memref type)     │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  For Each Memref Root:                  │
        │    handleMemref(root)                   │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  1. Validate (type, aliasing)           │
        │  2. Create to_tensor operation          │
        │  3. Sort users by execution order       │
        │  4. Transform each user                 │
        │  5. Convert back to memref              │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Cleanup: Remove Dead Allocations       │
        │    debufferizationAllocaRemoval         │
        └─────────────────────────────────────────┘
```

---

## Algorithm Flowchart

### Main Algorithm: `matchAndRewrite(func::FuncOp)`

```
                        START
                          │
                          ▼
            ┌──────────────────────────┐
            │  Collect Memref Roots:   │
            │  • AllocaOps             │
            │  • AllocOps              │
            │  • Function Arguments    │
            └──────────────────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │  For each root:          │
            │    handleMemref(root) ───┼──┐
            └──────────────────────────┘  │
                          │               │
                          ▼               │
                ┌─────────────────┐       │
                │  Any Success?   │       │
                └─────────────────┘       │
                    │           │         │
                   Yes         No         │
                    │           │         │
                    ▼           ▼         │
               SUCCESS      FAILURE       │
                                          │
                          ┌───────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │       handleMemref(memVal)              │
        └─────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │  Validation:                            │
        │  ✓ Is MemRefType?                       │
        │  ✓ Check aliasing (noalias preferred)   │
        │  ✓ Not already debufferized?            │
        └─────────────────────────────────────────┘
                          │
                    ┌─────┴─────┐
                    │           │
                  Valid     Invalid
                    │           │
                    ▼           ▼
        ┌───────────────┐   FAILURE
        │  to_tensor    │
        │  currentTensor│
        └───────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────┐
        │  sortedUsers = getSortedUsers(memVal)   │
        │  (Order by program execution)           │
        └─────────────────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────┐
        │  For each user in sortedUsers:          │
        └─────────────────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────┐
        │  Is currentTensor in right scope?       │
        │  (Check common ancestor region)         │
        └─────────────────────────────────────────┘
                    │
              ┌─────┴─────┐
              │           │
             No          Yes
              │           │
              ▼           │
    ┌──────────────────┐  │
    │  Propagate Value │  │
    │  Through Regions │  │
    │  • scf.for       │  │
    │  • scf.if        │  │
    └──────────────────┘  │
              │           │
              └─────┬─────┘
                    │
                    ▼
        ┌─────────────────────────────────────────┐
        │  What type of user?                     │
        └─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
  linalg.generic  memref.     Other
                  subview    (SKIP!)
        │           │
        │           │
        ▼           ▼
  Transform    Convert to
  to Tensor    extract_slice
        │           │
        └─────┬─────┘
              │
              ▼
    ┌──────────────────────┐
    │ Update currentTensor │
    │ (to result/output)   │
    └──────────────────────┘
              │
              ▼
    ┌──────────────────────┐
    │  All users done?     │
    └──────────────────────┘
              │
        ┌─────┴─────┐
        │           │
       No          Yes
        │           │
        │           ▼
        │  ┌──────────────────────┐
        │  │  Final propagation   │
        │  │  to outer scope      │
        │  └──────────────────────┘
        │           │
        │           ▼
        │  ┌──────────────────────┐
        │  │  Was tensor changed? │
        │  └──────────────────────┘
        │           │
        │     ┌─────┴─────┐
        │     │           │
        │    Yes         No
        │     │           │
        │     ▼           │
        │  to_memref      │
        │  memref.copy    │
        │     │           │
        │     └─────┬─────┘
        │           │
        └───────────┤
                    ▼
                 SUCCESS
```

---

## Detailed Algorithm

### Phase 1: Memref Root Collection

```cpp
void matchAndRewrite(func::FuncOp funcOp, PatternRewriter &rewriter) {
  SmallVector<memref::AllocaOp> listOfAllocaOps;
  SmallVector<memref::AllocOp> listOfAllocOps;
  
  // Collect all stack allocations
  funcOp.walk([&](memref::AllocaOp alloca) {
    listOfAllocaOps.push_back(alloca);
  });
  
  // Collect all heap allocations
  funcOp.walk([&](memref::AllocOp alloc) {
    listOfAllocOps.push_back(alloc);
  });
  
  // Process each root
  for (auto alloca : listOfAllocaOps) {
    anySuccess |= succeeded(handleMemref(alloca));
  }
  
  for (auto alloc : listOfAllocOps) {
    anySuccess |= succeeded(handleMemref(alloc));
  }
  
  // Process function arguments
  for (auto arg : funcOp.getArguments()) {
    anySuccess |= succeeded(handleMemref(arg));
  }
}
```

**Why these three types?**
- They represent **independent memory objects**
- No aliasing concerns (or marked as noalias)
- Safe to transform to tensors

---

### Phase 2: handleMemref - Core Transformation

#### Step 1: Validation

```cpp
LogicalResult handleMemref(Value memVal) {
  // 1. Type check
  if (!memVal.getType().isa<MemRefType>())
    return failure();
  
  // 2. Aliasing analysis
  bool isNoalias = false;
  if (auto allocaOp = memVal.getDefiningOp<memref::AllocaOp>())
    isNoalias = true;  // Stack allocations are local
  else if (auto ba = dyn_cast<BlockArgument>(memVal))
    isNoalias = checkNoaliasAttr(ba);
  
  // 3. Check if already debufferized
  auto sortedUsers = getSortedUsers(memVal);
  if (!sortedUsers.empty() && isa<ToTensorOp>(sortedUsers[0]))
    return failure();  // Already in tensor form
```

#### Step 2: Create Initial Tensor

```cpp
  // Convert memref to tensor
  MemRefType memrefType = getMemRefType(memVal);
  auto tensorType = RankedTensorType::get(
      memrefType.getShape(), 
      memrefType.getElementType());
  
  auto toTensorOp = rewriter.create<bufferization::ToTensorOp>(
      memVal.getLoc(), tensorType, memVal);
  
  Value currentTensor = toTensorOp;
```

**Transformation:**
```mlir
%memref = memref.alloca() : memref<10xf32>
// Becomes:
%memref = memref.alloca() : memref<10xf32>
%tensor = bufferization.to_tensor %memref : memref<10xf32>
```

#### Step 3: Sort Users

```cpp
  auto sortedUsers = getSortedUsers(memVal);
  std::vector<Operation*> expandedUserList(sortedUsers);
```

**Critical:** Users must be processed in **program execution order** to maintain correctness.

#### Step 4: Process Each User

```cpp
  llvm::DenseMap<Operation*, opTuple> opResultMap;
  
  for (auto user : sortedUsers) {
    // === SCOPE MANAGEMENT ===
    // Find common ancestor region
    auto commonRegion = findCommonAncestorRegion(
        currentTensor.getDefiningOp(), user);
    
    // Collect regions to propagate through
    SmallVector<Region*> regions;
    for (Region* r = currentTensor.getParentRegion(); 
         r != commonRegion; 
         r = r->getParentOp()->getParentRegion()) {
      regions.push_back(r);
    }
    
    // Propagate value if needed
    if (!regions.empty()) {
      propagateValueThroughRegion(currentTensor, regions, 
                                  expandedUserList, opResultMap, rewriter);
    }
```

**Why is this needed?**

Example where propagation is required:
```mlir
%tensor = ...  // Outer scope
scf.for %i = 0 to 10 {
  linalg.generic outs(%tensor) { ... }  // Inner scope - needs propagation!
}
```

The tensor is defined in outer scope but used in inner scope. We need to:
1. Add it as an `iter_arg` to the loop
2. Update uses inside to reference the block argument
3. Yield it at the end

---

### Phase 3: User Type Handling

#### Case A: `linalg.generic` Operations

```cpp
    if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
      // Replace memref operands with tensor operands
      SmallVector<Value> newInputs;
      for (auto input : genericOp.getInputs()) {
        newInputs.push_back(input == memVal ? currentTensor : input);
      }
      
      SmallVector<Value> newOutputs;
      SmallVector<Type> resultTypes;
      int newCurrentTensorIndex = -1;
      
      for (auto output : genericOp.getOutputs()) {
        newOutputs.push_back(output == memVal ? currentTensor : output);
        resultTypes.push_back(output == memVal ? currentTensor.getType() 
                                                : output.getType());
        if (output == memVal) {
          newCurrentTensorIndex = index;  // Track our tensor
        }
        index++;
      }
      
      // Create new tensor-based linalg.generic
      auto newGenericOp = rewriter.create<linalg::GenericOp>(
          genericOp.getLoc(), 
          resultTypes,     // Returns tensors!
          newInputs,       // Tensor inputs
          newOutputs,      // Tensor outputs (init tensors)
          genericOp.getIndexingMaps(), 
          genericOp.getIteratorTypes()
      );
      
      // Clone computation region
      rewriter.cloneRegionBefore(genericOp.getRegion(), 
                                 newGenericOp.getRegion(), ...);
      
      // Update currentTensor to result
      if (newCurrentTensorIndex != -1) {
        currentTensor = newGenericOp.getResult(newCurrentTensorIndex);
        opResultMap[newGenericOp] = std::make_tuple(currentTensor, prevTensor);
      }
      
      // Delete old operation
      rewriter.eraseOp(genericOp);
    }
```

**Transformation:**
```mlir
// BEFORE
linalg.generic ins(%A : memref<10xf32>) 
               outs(%B : memref<10xf32>) {
  ^bb0(%a: f32, %b: f32):
    %c = arith.addf %a, %b : f32
    linalg.yield %c : f32
}

// AFTER
%result = linalg.generic ins(%A_tensor : tensor<10xf32>) 
                         outs(%B_tensor : tensor<10xf32>) -> tensor<10xf32> {
  ^bb0(%a: f32, %b: f32):
    %c = arith.addf %a, %b : f32
    linalg.yield %c : f32
}
```

#### Case B: `memref.subview` Operations

```cpp
    else if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
      if (subviewOp.getSource() == memVal) {
        // Convert to tensor.extract_slice
        auto extractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
            subviewOp.getLoc(), 
            currentTensor,  // Tensor input
            subviewOp.getOffsets(), 
            subviewOp.getSizes(), 
            subviewOp.getStrides()
        );
      }
    }
```

**Transformation:**
```mlir
// BEFORE
%view = memref.subview %base[0][10][1] : memref<100xf32> to memref<10xf32>

// AFTER
%view = tensor.extract_slice %base_tensor[0][10][1] : 
        tensor<100xf32> to tensor<10xf32>
```

#### Case C: Unknown Operations (⚠️ THE PROBLEM!)

```cpp
    else {
      // Skip unknown users
      // THIS IS WHERE polygeist.submap IS SKIPPED!
    }
  } // End user loop
```

**Why is this a problem?**
- `polygeist.submap` operations are not handled
- They create intermediate memref views
- `linalg.generic` operations that consume these views are never reached
- **This blocks debufferization of strided operations!**

---

### Phase 4: Final Conversion

```cpp
  // Final propagation to outer scope
  propagateValueThroughRegion(currentTensor, regions, 
                              expandedUserList, opResultMap, rewriter);
  
  // Convert tensor back to memref (if modified)
  if (currentTensor != toTensorOp) {
    auto toMemrefOp = rewriter.create<bufferization::ToMemrefOp>(
        memVal.getLoc(), memrefType, currentTensor);
    
    auto copyOp = rewriter.create<memref::CopyOp>(
        memVal.getLoc(), toMemrefOp, memVal);
  }
  
  return success();
}
```

**Final IR:**
```mlir
%tensor = bufferization.to_tensor %memref : memref<?xf32>
// ... tensor transformations ...
%final_tensor = linalg.generic ... -> tensor<?xf32>
%final_memref = bufferization.to_memref %final_tensor : memref<?xf32>
memref.copy %final_memref, %memref : memref<?xf32> to memref<?xf32>
```

---

## Key Data Structures

### 1. `opTuple` - Operation Result Tracking

```cpp
using opTuple = std::tuple<Value, Value>;
// First:  result tensor from this operation
// Second: init tensor (input to this operation)

llvm::DenseMap<Operation*, opTuple> opResultMap;
```

**Purpose:** Track transformations for region propagation

**Example:**
```mlir
// Original:
%init = tensor.empty() : tensor<10xf32>
%result = linalg.generic outs(%init) -> tensor<10xf32> { ... }

// Stored in map:
opResultMap[genericOp] = {%result, %init}
```

**Why needed?** When propagating through `scf.for` loops, we need to know:
- What tensor to use as `iter_arg` (init tensor)
- What tensor the loop produces (result tensor)

### 2. `expandedUserList` - Dynamic User Tracking

```cpp
std::vector<Operation*> expandedUserList(sortedUsers);
```

**Purpose:** Track operations as they're replaced

**Example:**
```cpp
// Before transformation
expandedUserList = [oldForOp, genericOp1, genericOp2]

// After transforming oldForOp
expandedUserList = [newForOp, genericOp1, genericOp2]
//                   ^^^^^^^^^ Replaced!

// Used by propagateValueThroughRegion to find the right newForOp
```

### 3. `sortedUsers` - Execution Order

```cpp
std::vector<Operation*> sortedUsers = getSortedUsers(memVal);
```

**Purpose:** Process users in program execution order

**Why critical?** SSA form requires definitions before uses.

---

## Helper Functions

### 1. `getSortedUsers(Value val)`

```cpp
std::vector<Operation*> getSortedUsers(Value val) {
  std::vector<Operation*> users;
  for (Operation *user : val.getUsers()) {
    // Deduplicate
    if (std::find(users.begin(), users.end(), user) == users.end())
      users.push_back(user);
  }
  
  // Sort by program order
  std::sort(users.begin(), users.end(), 
           [](Operation *a, Operation *b) {
             return comesBefore(a, b);
           });
  
  return users;
}
```

**Returns:** All users in **execution order**

### 2. `comesBefore(Operation *a, Operation *b)`

**Purpose:** Determine if operation `a` executes before `b`

**Algorithm:**
```
if a == b:
  return false

if a is ancestor of b:
  return true  // a surrounds b, so a starts first

if b is ancestor of a:
  return false  // b surrounds a, so b starts first

// Find common ancestor and compare positions
commonAncestor = findCommonParent(a, b)
compare positions within commonAncestor's regions
```

**Handles:**
- Same block: `a->isBeforeInBlock(b)`
- Different blocks: compare block order
- Different regions: compare region order
- Nested hierarchies: recursive comparison

### 3. `findCommonAncestorRegion(Operation *a, Operation *b)`

```cpp
Region* findCommonAncestorRegion(Operation* a, Operation* b) {
  DenseMap<Region*, size_t> regionCounts;
  
  // Walk up from a, marking all ancestor regions
  Operation* currentOp = a;
  while (Region* region = currentOp->getParentRegion()) {
    regionCounts[region]++;
    currentOp = region->getParentOp();
  }
  
  // Walk up from b, find first common region
  currentOp = b;
  while (Region* region = currentOp->getParentRegion()) {
    if (regionCounts.count(region))
      return region;  // Found common ancestor!
    currentOp = region->getParentOp();
  }
  
  return nullptr;
}
```

**Returns:** Innermost region containing both operations

**Example:**
```mlir
func.func @example() {              // Region 0
  %tensor = ...
  scf.for %i = 0 to 10 {            // Region 1
    scf.if %cond {                  // Region 2
      linalg.generic outs(%tensor)  // Operation b
    }
  }
}
// Operation a: %tensor definition (in Region 0)
// Operation b: linalg.generic (in Region 2)
// Common ancestor: Region 0
```

### 4. `findUsersInRegion(Value val, Region &region, ...)`

```cpp
void findUsersInRegion(Value value, Region& region, 
                       SmallVectorImpl<Operation*>& users) {
  for (Block& block : region) {
    for (Operation& op : block) {
      // Check if op uses value
      for (Value operand : op.getOperands()) {
        if (operand == value) {
          users.push_back(&op);
          break;
        }
      }
      
      // Recursively search sub-regions
      for (Region& subRegion : op.getRegions()) {
        findUsersInRegion(value, subRegion, users);
      }
    }
  }
}
```

**Purpose:** Find all operations in a region that use a specific value

**Used by:** `propagateValueThroughRegion` to find which operations need updating

---

## Region Propagation

This is the most complex part of the algorithm. When a tensor is defined in one scope but used in another, we need to **thread it through** the intermediate regions.

### Flowchart: `propagateValueThroughRegion`

```
                Input: currentValue, regions[], expandedUserList, opResultMap
                          │
                          ▼
              ┌────────────────────────────┐
              │  For each region:          │
              └────────────────────────────┘
                          │
                          ▼
              ┌────────────────────────────┐
              │  Find init tensor:         │
              │  • First use in region     │
              │  • From opResultMap        │
              └────────────────────────────┘
                          │
                          ▼
              ┌────────────────────────────┐
              │  What is parent op?        │
              └────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
        ┌─────────┐           ┌─────────┐
        │ scf.for │           │ scf.if  │
        └─────────┘           └─────────┘
              │                       │
              ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │ Add to iter_args │    │ Add to results   │
    └──────────────────┘    └──────────────────┘
              │                       │
              ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │ Create new loop  │    │ Create new if    │
    │ with extra arg   │    │ with extra result│
    └──────────────────┘    └──────────────────┘
              │                       │
              ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │ Update users     │    │ Update yields    │
    │ inside to use    │    │ • then: current  │
    │ block argument   │    │ • else: init     │
    └──────────────────┘    └──────────────────┘
              │                       │
              ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │ Update yield to  │    │ Update current   │
    │ return tensor    │    │ to if result     │
    └──────────────────┘    └──────────────────┘
              │                       │
              ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │ Update current   │    │ Store in         │
    │ to loop result   │    │ opResultMap      │
    └──────────────────┘    └──────────────────┘
              │                       │
              └───────────┬───────────┘
                          ▼
              ┌────────────────────────────┐
              │  Store in opResultMap      │
              │  Update expandedUserList   │
              └────────────────────────────┘
```

### Case A: `scf.for` Loops

**Problem:**
```mlir
%tensor = ...  // Outer scope
scf.for %i = 0 to 10 {
  linalg.generic outs(%tensor) { ... }  // ❌ Captures outer variable
}
```

**Solution: Add as `iter_arg`**
```mlir
%tensor = ...
%result = scf.for %i = 0 to 10 iter_args(%arg = %tensor) -> tensor<?xf32> {
  %updated = linalg.generic outs(%arg) { ... }
  scf.yield %updated : tensor<?xf32>
}
%tensor_after = %result
```

**Implementation:**
```cpp
if (auto prevFor = dyn_cast<scf::ForOp>(parentOp)) {
  // 1. Find which operations use initTensor inside the loop
  findUsersInRegion(initTensor, *region, initOpUsers);
  
  // 2. Add tensor to loop's iter_args
  SmallVector<Value> newInitOperands = prevFor.getInitArgs();
  newInitOperands.push_back(initTensor);
  
  // 3. Create new loop with extra iter_arg
  scf::ForOp newLoop = rewriter.create<scf::ForOp>(
      prevFor.getLoc(),
      prevFor.getLowerBound(),
      prevFor.getUpperBound(),
      prevFor.getStep(),
      newInitOperands  // Now includes our tensor
  );
  
  // 4. Transfer operations to new loop
  Block *newBlock = &newLoop.getRegion().front();
  Block *originalBlock = &prevFor.getRegion().front();
  newBlock->getOperations().splice(newBlock->end(),
                                   originalBlock->getOperations());
  
  // 5. Update uses of initTensor to use the new block argument
  Value newIterArg = newLoop.getRegionIterArg(numArgs - 2);
  for (auto initOpUser : initOpUsers) {
    for (auto &en : llvm::enumerate(initOpUser->getOperands())) {
      if (en.value() == initTensor) {
        initOpUser->setOperand(en.index(), newIterArg);
      }
    }
  }
  
  // 6. Update yield to return the tensor
  auto yieldOp = newBlock->getTerminator();
  SmallVector<Value> newYieldValues = yieldOp->getOperands();
  newYieldValues.push_back(currentValue);
  rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newYieldValues);
  
  // 7. Update currentValue to be the loop result
  currentValue = newLoop.getResults().back();
  
  // 8. Track this transformation
  opResultMap[newLoop] = std::make_tuple(currentValue, initTensor);
  
  // 9. Erase old loop
  rewriter.eraseOp(prevFor);
}
```

### Case B: `scf.if` Operations

**Problem:**
```mlir
%tensor = ...  // Outer scope
scf.if %cond {
  linalg.generic outs(%tensor) { ... }  // ❌ Captures outer variable
}
```

**Solution: Add as result**
```mlir
%tensor = ...
%result = scf.if %cond -> tensor<?xf32> {
  %updated = linalg.generic outs(%tensor) { ... }
  scf.yield %updated : tensor<?xf32>
} else {
  scf.yield %tensor : tensor<?xf32>  // Unchanged in else
}
%tensor_after = %result
```

**Implementation:**
```cpp
if (auto prevIf = dyn_cast<scf::IfOp>(parentOp)) {
  // 1. Add tensor to result types
  SmallVector<Type> newResultTypes = prevIf.getResultTypes();
  newResultTypes.push_back(currentValue.getType());
  
  // 2. Update then region yield
  SmallVector<Value> thenYieldValues = prevIf.thenYield().getOperands();
  thenYieldValues.push_back(currentValue);
  
  // 3. Update else region yield (or use initTensor if no modification)
  SmallVector<Value> elseYieldValues;
  if (!prevIf.getElseRegion().empty()) {
    elseYieldValues = prevIf.elseYield().getOperands();
  }
  elseYieldValues.push_back(initTensor);  // Unchanged in else
  
  // 4. Create new if with updated yields
  auto newIf = rewriter.create<scf::IfOp>(
      prevIf.getLoc(),
      newResultTypes,
      prevIf.getCondition(),
      true  // withElseRegion
  );
  
  // 5. Transfer regions
  newIf.getThenRegion().takeBody(prevIf.getThenRegion());
  if (!prevIf.getElseRegion().empty())
    newIf.getElseRegion().takeBody(prevIf.getElseRegion());
  
  // 6. Update yield operations
  rewriter.setInsertionPointToEnd(newIf.thenBlock());
  rewriter.replaceOpWithNewOp<scf::YieldOp>(newIf.thenYield(), thenYieldValues);
  
  rewriter.setInsertionPointToEnd(newIf.elseBlock());
  if (!prevIf.getElseRegion().empty())
    rewriter.replaceOpWithNewOp<scf::YieldOp>(newIf.elseYield(), elseYieldValues);
  else
    rewriter.create<scf::YieldOp>(newIf.getLoc(), elseYieldValues);
  
  // 7. Update currentValue to if result
  currentValue = newIf->getResult(newIf->getNumResults() - 1);
  
  // 8. Track this transformation
  opResultMap[newIf] = std::make_tuple(currentValue, initTensor);
  
  // 9. Erase old if
  rewriter.eraseOp(prevIf);
}
```

---

## Example Transformations

### Example 1: Simple Transformation

**Input:**
```mlir
func.func @simple(%A: memref<10xf32>, %B: memref<10xf32>) {
  %tmp = memref.alloca() : memref<10xf32>
  
  linalg.generic ins(%A : memref<10xf32>) 
                 outs(%tmp : memref<10xf32>) {
  ^bb0(%a: f32, %out: f32):
    %c = arith.mulf %a, %a : f32
    linalg.yield %c : f32
  }
  
  linalg.generic ins(%tmp : memref<10xf32>) 
                 outs(%B : memref<10xf32>) {
  ^bb0(%a: f32, %out: f32):
    %c = arith.addf %a, %a : f32
    linalg.yield %c : f32
  }
  
  return
}
```

**Step-by-Step Transformation:**

1. **Identify root:** `%tmp = memref.alloca()`

2. **Create to_tensor:**
```mlir
%tmp = memref.alloca() : memref<10xf32>
%tmp_tensor = bufferization.to_tensor %tmp : memref<10xf32>
```

3. **Process first generic:**
```mlir
%A_tensor = bufferization.to_tensor %A : memref<10xf32>
%init1 = tensor.empty() : tensor<10xf32>
%result1 = linalg.generic ins(%A_tensor : tensor<10xf32>) 
                          outs(%init1 : tensor<10xf32>) -> tensor<10xf32> {
^bb0(%a: f32, %out: f32):
  %c = arith.mulf %a, %a : f32
  linalg.yield %c : f32
}
// currentTensor = %result1
```

4. **Process second generic:**
```mlir
%B_tensor = bufferization.to_tensor %B : memref<10xf32>
%result2 = linalg.generic ins(%result1 : tensor<10xf32>) 
                          outs(%B_tensor : tensor<10xf32>) -> tensor<10xf32> {
^bb0(%a: f32, %out: f32):
  %c = arith.addf %a, %a : f32
  linalg.yield %c : f32
}
// currentTensor = %result2
```

5. **Convert back:**
```mlir
%B_memref = bufferization.to_memref %result2 : memref<10xf32>
memref.copy %B_memref, %B : memref<10xf32> to memref<10xf32>
```

**Output:**
```mlir
func.func @simple(%A: memref<10xf32>, %B: memref<10xf32>) {
  %tmp = memref.alloca() : memref<10xf32>
  %tmp_tensor = bufferization.to_tensor %tmp : memref<10xf32>
  
  %A_tensor = bufferization.to_tensor %A : memref<10xf32>
  %init1 = tensor.empty() : tensor<10xf32>
  %result1 = linalg.generic ins(%A_tensor : tensor<10xf32>) 
                            outs(%init1 : tensor<10xf32>) -> tensor<10xf32> {
  ^bb0(%a: f32, %out: f32):
    %c = arith.mulf %a, %a : f32
    linalg.yield %c : f32
  }
  
  %B_tensor = bufferization.to_tensor %B : memref<10xf32>
  %result2 = linalg.generic ins(%result1 : tensor<10xf32>) 
                            outs(%B_tensor : tensor<10xf32>) -> tensor<10xf32> {
  ^bb0(%a: f32, %out: f32):
    %c = arith.addf %a, %a : f32
    linalg.yield %c : f32
  }
  
  %B_memref = bufferization.to_memref %result2 : memref<10xf32>
  memref.copy %B_memref, %B : memref<10xf32> to memref<10xf32>
  
  return
}
```

**Benefits:**
- Now `%result1` and `%result2` can be fused by later passes
- Data flow is explicit

---

### Example 2: Loop with Region Propagation

**Input:**
```mlir
func.func @loop(%A: memref<10xf32>, %n: index) {
  %tmp = memref.alloca() : memref<10xf32>
  
  scf.for %i = %c0 to %n step %c1 {
    linalg.generic ins(%A : memref<10xf32>) 
                   outs(%tmp : memref<10xf32>) {
    ^bb0(%a: f32, %out: f32):
      %c = arith.addf %out, %a : f32
      linalg.yield %c : f32
    }
  }
  
  return
}
```

**Transformation Steps:**

1. **Create to_tensor:**
```mlir
%tmp = memref.alloca() : memref<10xf32>
%tmp_tensor = bufferization.to_tensor %tmp : memref<10xf32>
```

2. **Detect scope issue:**
   - `linalg.generic` is inside `scf.for`
   - `%tmp_tensor` is outside `scf.for`
   - Need propagation!

3. **Propagate through scf.for:**
```mlir
%result = scf.for %i = %c0 to %n step %c1 
          iter_args(%iter_tensor = %tmp_tensor) -> tensor<10xf32> {
  %A_tensor = bufferization.to_tensor %A : memref<10xf32>
  %updated = linalg.generic ins(%A_tensor : tensor<10xf32>) 
                            outs(%iter_tensor : tensor<10xf32>) -> tensor<10xf32> {
  ^bb0(%a: f32, %out: f32):
    %c = arith.addf %out, %a : f32
    linalg.yield %c : f32
  }
  scf.yield %updated : tensor<10xf32>
}
```

4. **Convert back:**
```mlir
%tmp_memref = bufferization.to_memref %result : memref<10xf32>
memref.copy %tmp_memref, %tmp : memref<10xf32> to memref<10xf32>
```

**Output:**
```mlir
func.func @loop(%A: memref<10xf32>, %n: index) {
  %tmp = memref.alloca() : memref<10xf32>
  %tmp_tensor = bufferization.to_tensor %tmp : memref<10xf32>
  
  %result = scf.for %i = %c0 to %n step %c1 
            iter_args(%iter_tensor = %tmp_tensor) -> tensor<10xf32> {
    %A_tensor = bufferization.to_tensor %A : memref<10xf32>
    %updated = linalg.generic ins(%A_tensor : tensor<10xf32>) 
                              outs(%iter_tensor : tensor<10xf32>) -> tensor<10xf32> {
    ^bb0(%a: f32, %out: f32):
      %c = arith.addf %out, %a : f32
      linalg.yield %c : f32
    }
    scf.yield %updated : tensor<10xf32>
  }
  
  %tmp_memref = bufferization.to_memref %result : memref<10xf32>
  memref.copy %tmp_memref, %tmp : memref<10xf32> to memref<10xf32>
  
  return
}
```

**Key change:** The tensor is now threaded through the loop as an `iter_arg`, making the accumulation explicit!

---

### Example 3: Nested Conditionals

**Input:**
```mlir
func.func @conditional(%A: memref<10xf32>, %cond1: i1, %cond2: i1) {
  %tmp = memref.alloca() : memref<10xf32>
  
  scf.if %cond1 {
    scf.if %cond2 {
      linalg.generic ins(%A : memref<10xf32>) 
                     outs(%tmp : memref<10xf32>) {
      ^bb0(%a: f32, %out: f32):
        linalg.yield %a : f32
      }
    }
  }
  
  return
}
```

**Output (after propagation):**
```mlir
func.func @conditional(%A: memref<10xf32>, %cond1: i1, %cond2: i1) {
  %tmp = memref.alloca() : memref<10xf32>
  %tmp_tensor = bufferization.to_tensor %tmp : memref<10xf32>
  
  %result = scf.if %cond1 -> tensor<10xf32> {
    %inner_result = scf.if %cond2 -> tensor<10xf32> {
      %A_tensor = bufferization.to_tensor %A : memref<10xf32>
      %updated = linalg.generic ins(%A_tensor : tensor<10xf32>) 
                                outs(%tmp_tensor : tensor<10xf32>) -> tensor<10xf32> {
      ^bb0(%a: f32, %out: f32):
        linalg.yield %a : f32
      }
      scf.yield %updated : tensor<10xf32>
    } else {
      scf.yield %tmp_tensor : tensor<10xf32>
    }
    scf.yield %inner_result : tensor<10xf32>
  } else {
    scf.yield %tmp_tensor : tensor<10xf32>
  }
  
  %tmp_memref = bufferization.to_memref %result : memref<10xf32>
  memref.copy %tmp_memref, %tmp : memref<10xf32> to memref<10xf32>
  
  return
}
```

**Key:** The tensor is propagated through **both** nested `scf.if` operations!

---

## Current Limitations

### 1. ⚠️ **Skips `polygeist.submap` Operations**

**Problem:**
```mlir
%x_view = polygeist.submap(%x, %stride, %size) 
          <{map = affine_map<(d0)[s0] -> (d0 * s0)>}>
          : (memref<?xf32>, index, index) -> memref<?xf32>

linalg.generic ins(%x_view : memref<?xf32>) ...  // ❌ NEVER REACHED!
```

**Why:**
- `polygeist.submap` creates an intermediate memref view
- The algorithm only looks at direct users of the base memref
- `linalg.generic` consuming the submap is never encountered
- **This blocks all strided BLAS operations!**

**Impact:** High - affects all Level 1 BLAS with stride parameters

**See:** `RaiseToLinalg_Issues.md` - Issue #3

### 2. **Limited Operation Support**

Currently only handles:
- `linalg.generic`
- `memref.subview` → `tensor.extract_slice`

**Not handled:**
- `polygeist.submap`
- Other linalg ops (`linalg.matmul`, `linalg.conv`, etc.)
- Custom operations

### 3. **No Multi-Value Propagation**

**Limitation:** Only propagates one tensor at a time through regions

**Example that could fail:**
```mlir
%A_tensor = ...
%B_tensor = ...
scf.for %i = ... {
  linalg.generic ins(%A_tensor) outs(%B_tensor) ...
}
```

The algorithm would try to propagate `%A_tensor` and `%B_tensor` separately, which could lead to issues.

### 4. **Aliasing Analysis is Basic**

```cpp
bool isNoalias = false;
if (auto allocaOp = memVal.getDefiningOp<memref::AllocaOp>())
  isNoalias = true;  // Assumes all allocas are noalias
```

**Issue:** Doesn't perform deep aliasing analysis, may miss optimization opportunities or transform incorrectly.

### 5. **No Partial Debufferization**

**Current behavior:** All-or-nothing per memref root

**Would be useful:** Debufferize only certain uses, leave others as memref

---

## Future Work

### Priority 1: Handle `polygeist.submap` ⭐⭐⭐

**Proposal:** Add case in user processing loop:

```cpp
else if (auto submapOp = dyn_cast<polygeist::SubmapOp>(user)) {
  // Extract affine map and operands
  AffineMap map = submapOp.getMap();
  Value stride = submapOp.getStride();
  Value size = submapOp.getSize();
  
  // Check if stride is constant
  if (auto constStride = getConstantIntValue(stride)) {
    // Use tensor.extract_slice with static stride
    auto sliceOp = rewriter.create<tensor::ExtractSliceOp>(
        submapOp.getLoc(), currentTensor, 
        /*offsets=*/..., /*sizes=*/..., /*strides=*/*constStride);
    currentTensor = sliceOp;
  } else {
    // Create scf.for gather loop
    auto gatherLoop = createGatherLoop(currentTensor, stride, size);
    currentTensor = gatherLoop;
  }
  
  // Recursively process users of submap
  auto submapUsers = getSortedUsers(submapOp.getResult());
  for (auto submapUser : submapUsers) {
    // Process linalg.generic that consumes the submap
    // ...
  }
}
```

**See:** `saxpy_debufferized_example.mlir` for detailed transformation examples

### Priority 2: Extend to Other Linalg Ops

**Current:** Only `linalg.generic`

**Extend to:**
- `linalg.matmul`
- `linalg.conv`
- `linalg.fill`
- `linalg.copy`
- Custom named linalg ops

**Implementation:** Similar pattern to `linalg.generic` handling

### Priority 3: Better Aliasing Analysis

**Current:** Basic checks on operation type

**Improve:**
- Use MLIR's aliasing analysis framework
- Track memory effects more precisely
- Handle partial aliasing

### Priority 4: Multi-Value Propagation

**Goal:** Propagate multiple tensors through regions simultaneously

**Challenge:** Tracking dependencies between multiple tensors

**Benefit:** Handle more complex kernels with multiple outputs

### Priority 5: Partial Debufferization

**Goal:** Debufferize only profitable uses

**Use case:** Some uses are better left as memref (e.g., escaping pointers)

**Implementation:** Cost model + selective transformation

---

## Summary

The `LinalgDebufferize` pass is a sophisticated transformation that:

✅ **Converts memref-based linalg ops to tensor-based ops**  
✅ **Handles complex control flow (loops, conditionals)**  
✅ **Threads values through nested regions**  
✅ **Enables high-level linalg optimizations**

❌ **Currently skips `polygeist.submap` operations**  
❌ **Limited to `linalg.generic` and `memref.subview`**  
❌ **Basic aliasing analysis**

**Key Innovation:** Region propagation algorithm that automatically updates `scf.for`/`scf.if` operations to thread tensors through their regions.

**Main Limitation:** Doesn't handle strided memory access patterns created by `polygeist.submap`, blocking debufferization of BLAS operations.

**Next Steps:** Extend the pass to handle `polygeist.submap` by materializing gather/scatter operations (see Issue #3 in `RaiseToLinalg_Issues.md`).

---

## References

- **Implementation:** `lib/polygeist/Passes/LinalgDebufferize.cpp`
- **Related Issues:** `RaiseToLinalg_Issues.md` - Issue #3
- **Example Transformations:** `saxpy_debufferized_example.mlir`
- **Test Cases:** `test/polygeist-opt/` (TODO: add debufferize tests)
- **MLIR Linalg Dialect:** https://mlir.llvm.org/docs/Dialects/Linalg/
- **MLIR Bufferization:** https://mlir.llvm.org/docs/Bufferization/

---

**Document Version:** 1.0  
**Last Updated:** October 17, 2025  
**Maintained by:** Polygeist Team

