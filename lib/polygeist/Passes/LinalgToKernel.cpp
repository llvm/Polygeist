//===- LinalgToKernel.cpp - Pattern to match linalg.generic with kernel.defn ------===//
//
// This file implements a pattern to rewrite linalg.generic operations to kernel
// operations by matching against patterns defined in kernel.defn_collection.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Debug.h"
#include "polygeist/Kernel/KernelDialect.h"
#include "polygeist/Kernel/KernelOps.h"
#include "polygeist/Passes/Passes.h"

#include <stack>
#include <set>
#include <functional>

#define DEBUG_TYPE "linalg-to-kernel"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::polygeist;
using namespace mlir::polygeist::kernel;

namespace {

static bool areKernelOperandTypesCompatible(Type actual, Type specification) {
  if (actual == specification)
    return true;
  if (isa<RankedTensorType>(actual) != isa<RankedTensorType>(specification) ||
      isa<MemRefType>(actual) != isa<MemRefType>(specification))
    return false;
  auto actualShaped = dyn_cast<ShapedType>(actual);
  auto specificationShaped = dyn_cast<ShapedType>(specification);
  if (!actualShaped || !specificationShaped)
    return false;
  return actualShaped.hasRank() && specificationShaped.hasRank() &&
         actualShaped.getRank() == specificationShaped.getRank() &&
         actualShaped.getElementType() == specificationShaped.getElementType();
}

static bool areGenericOperandTypesCompatible(GenericOp actual,
                                             GenericOp specification) {
  if (actual->getNumOperands() != specification->getNumOperands())
    return false;
  return llvm::all_of(llvm::zip(actual->getOperandTypes(),
                                specification->getOperandTypes()),
                      [](auto pair) {
                        return areKernelOperandTypesCompatible(
                            std::get<0>(pair), std::get<1>(pair));
                      });
}

// Structure to represent an operation node in the dependency graph
struct OpNode {
  Operation *op;
  StringRef opName;
  SmallVector<Type> operandTypes;
  SmallVector<Type> resultTypes;
  SmallVector<OpNode*> dependencies;  // Operations this depends on
  SmallVector<OpNode*> dependents;    // Operations that depend on this
  
  OpNode(Operation *operation) : op(operation) {
    if (operation) {
      // Regular operation node
      opName = operation->getName().getStringRef();
      for (Value operand : operation->getOperands()) {
        operandTypes.push_back(operand.getType());
      }
      for (Value result : operation->getResults()) {
        resultTypes.push_back(result.getType());
      }
    } else {
      // Special node for block arguments - will be set later
      opName = "block_arg";
    }
  }
  
  // Check if two nodes are structurally equivalent (same operation type and types)
  bool isEquivalentTo(const OpNode &other) const {
    return opName == other.opName && 
           operandTypes == other.operandTypes && 
           resultTypes == other.resultTypes;
  }
};

// Structure to represent a dependency graph for a region
struct DependencyGraph {
  SmallVector<std::unique_ptr<OpNode>> nodes;
  DenseMap<Operation*, OpNode*> opToNode;
  SmallVector<OpNode*> blockArgNodes;  // Special nodes for block arguments
  
  void buildFromRegion(Region &region) {
    // Process each block in the region
    for (Block &block : region.getBlocks()) {
      
      // Create pseudo-nodes for block arguments
      for (BlockArgument arg : block.getArguments()) {
        // Block arguments are represented as special nodes
        auto argNode = std::make_unique<OpNode>(nullptr);
        argNode->resultTypes.push_back(arg.getType());
        blockArgNodes.push_back(argNode.get());
        
        // Map the block argument value to this node for dependency tracking
        // We'll use a separate map for this
        nodes.push_back(std::move(argNode));
      }
      
      // Create nodes for each operation
      for (Operation &op : block.getOperations()) {
        auto node = std::make_unique<OpNode>(&op);
        OpNode *nodePtr = node.get();
        opToNode[&op] = nodePtr;
        nodes.push_back(std::move(node));
      }
      
      // Build dependency edges
      for (Operation &op : block.getOperations()) {
        OpNode *currentNode = opToNode[&op];
        
        // For each operand, find what it depends on
        for (Value operand : op.getOperands()) {
          if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
            // Depends on a block argument
            size_t argIndex = blockArg.getArgNumber();
            if (argIndex < blockArgNodes.size()) {
              OpNode *argNode = blockArgNodes[argIndex];
              currentNode->dependencies.push_back(argNode);
              argNode->dependents.push_back(currentNode);
            }
          } else if (Operation *definingOp = operand.getDefiningOp()) {
            // Depends on another operation
            if (opToNode.count(definingOp)) {
              OpNode *depNode = opToNode[definingOp];
              currentNode->dependencies.push_back(depNode);
              depNode->dependents.push_back(currentNode);
            }
          }
        }
      }
    }
  }
  
  // Get nodes in topological order (dependencies first)
  SmallVector<OpNode*> getTopologicalOrder() const {
    SmallVector<OpNode*> result;
    DenseSet<OpNode*> visited;
    
    std::function<void(OpNode*)> dfs = [&](OpNode* node) {
      if (visited.contains(node)) return;
      visited.insert(node);
      
      // Visit all dependencies first
      for (OpNode* dep : node->dependencies) {
        dfs(dep);
      }
      
      result.push_back(node);
    };
    
    // Start DFS from all nodes
    for (const auto &node : nodes) {
      dfs(node.get());
    }
    
    return result;
  }
};

// Enhanced region equivalence check using dependency graphs
bool areRegionsEquivalent(Region &first, Region &second, DenseMap<OpNode*, OpNode*> &nodeMapping, 
                         DenseMap<Operation*, Operation*> &operationMapping) {
  // Clear the output mappings
  nodeMapping.clear();
  operationMapping.clear();
  
  // Fast early checks before expensive graph construction
  
  // Check number of blocks
  if (first.getBlocks().size() != second.getBlocks().size()) {
    return false;
  }
  
  // Check each block's basic properties
  for (auto blockPair : llvm::zip(first.getBlocks(), second.getBlocks())) {
    Block &firstBlock = std::get<0>(blockPair);
    Block &secondBlock = std::get<1>(blockPair);
    
    // Check number of arguments
    if (firstBlock.getNumArguments() != secondBlock.getNumArguments()) {
      return false;
    }
    
    // Check argument types
    for (auto argPair : llvm::zip(firstBlock.getArguments(), secondBlock.getArguments())) {
      if (std::get<0>(argPair).getType() != std::get<1>(argPair).getType()) {
        return false;
      }
    }
    
    // Check number of operations
    if (firstBlock.getOperations().size() != secondBlock.getOperations().size()) {
      return false;
    }
  }
  
  // If basic checks pass, proceed with detailed graph-based analysis
  // Build dependency graphs for both regions
  DependencyGraph firstGraph, secondGraph;
  firstGraph.buildFromRegion(first);
  secondGraph.buildFromRegion(second);
  
  // Quick structural checks
  if (firstGraph.nodes.size() != secondGraph.nodes.size()) {
    return false;
  }
  
  if (firstGraph.blockArgNodes.size() != secondGraph.blockArgNodes.size()) {
    return false;
  }
  
  // Get topological orderings
  auto firstOrder = firstGraph.getTopologicalOrder();
  auto secondOrder = secondGraph.getTopologicalOrder();
  
  if (firstOrder.size() != secondOrder.size()) {
    return false;
  }
  
  // Compare nodes in topological order and build mapping
  for (size_t i = 0; i < firstOrder.size(); ++i) {
    OpNode *firstNode = firstOrder[i];
    OpNode *secondNode = secondOrder[i];
    
    // Check if the nodes are structurally equivalent
    if (!firstNode->isEquivalentTo(*secondNode)) {
      return false;
    }
    
    // Check if dependency structure matches
    if (firstNode->dependencies.size() != secondNode->dependencies.size()) {
      return false;
    }
    
    // Verify that dependencies map correctly
    for (size_t j = 0; j < firstNode->dependencies.size(); ++j) {
      OpNode *firstDep = firstNode->dependencies[j];
      OpNode *secondDep = secondNode->dependencies[j];
      
      // Check if we've established a mapping for these dependencies
      auto it = nodeMapping.find(firstDep);
      if (it != nodeMapping.end()) {
        if (it->second != secondDep) {
          return false; // Inconsistent mapping
        }
      } else {
        nodeMapping[firstDep] = secondDep;
      }
    }
    
    // Establish mapping for current nodes
    nodeMapping[firstNode] = secondNode;
    
    // Build the operation mapping directly from OpNode data while still valid
    if (firstNode->op && secondNode->op) {
      operationMapping[firstNode->op] = secondNode->op;
    }
  }

  return true;
}

// Helper to check if indexing maps are equivalent
bool areIndexingMapsEquivalent(ArrayAttr firstMaps, ArrayAttr secondMaps) {
  if (firstMaps.size() != secondMaps.size())
    return false;

  for (auto mapPair : llvm::zip(firstMaps, secondMaps)) {
    auto firstMap = std::get<0>(mapPair).cast<AffineMapAttr>().getValue();
    auto secondMap = std::get<1>(mapPair).cast<AffineMapAttr>().getValue();
    
    if (firstMap != secondMap)
      return false;
  }

  return true;
}

// Helper to check if iterator types are equivalent
bool areIteratorTypesEquivalent(ArrayAttr firstTypes, ArrayAttr secondTypes) {
  if (firstTypes.size() != secondTypes.size())
    return false;

  for (auto typePair : llvm::zip(firstTypes, secondTypes)) {
    auto firstType = std::get<0>(typePair).cast<linalg::IteratorTypeAttr>().getValue();
    auto secondType = std::get<1>(typePair).cast<linalg::IteratorTypeAttr>().getValue();
    
    if (firstType != secondType)
      return false;
  }

  return true;
}

// Helper function to find the corresponding value in actual IR for a kernel block argument
Value findCorrespondingValue(BlockArgument kernelArg, 
                            const DenseMap<Operation*, Operation*> &operationMapping,
                            GenericOp genericOp) {
  
  LLVM_DEBUG(llvm::dbgs() << "Finding corresponding value for kernel arg #" << kernelArg.getArgNumber() 
                          << " with type " << kernelArg.getType() << "\n");
  
  // First, check if this kernel argument is used as an operand to the linalg.generic itself
  // This handles tensor arguments that become ins/outs operands
  for (Operation *kernelUser : kernelArg.getUsers()) {
    LLVM_DEBUG(llvm::dbgs() << "Kernel arg used by: " << *kernelUser << "\n");
    
    // Check if the user is a linalg.generic operation
    if (auto kernelGeneric = dyn_cast<GenericOp>(kernelUser)) {
      LLVM_DEBUG(llvm::dbgs() << "Kernel arg is used by linalg.generic as operand\n");
      
      // Find which operand position kernelArg occupies in the kernel's linalg.generic
      size_t operandIndex = 0;
      for (Value operand : kernelGeneric->getOperands()) {
        if (operand == kernelArg) {
          LLVM_DEBUG(llvm::dbgs() << "Kernel arg is at operand index " << operandIndex 
                                  << " of kernel linalg.generic\n");
          
          // The corresponding operand in the actual linalg.generic should be at the same position
          if (operandIndex < genericOp->getNumOperands()) {
            Value actualOperand = genericOp->getOperand(operandIndex);
            LLVM_DEBUG(llvm::dbgs() << "Found corresponding actual operand: " << actualOperand << "\n");
            return actualOperand;
          } else {
            LLVM_DEBUG(llvm::dbgs() << "ERROR - operand index out of bounds in actual generic\n");
          }
          break;
        }
        operandIndex++;
      }
      
      // If we found a linalg.generic usage, we're done with this user
      break;
    }
  }
  
  // If we reach here, this might be a scalar argument used inside the region
  // For scalar arguments like %arg3, %arg4, use operation mapping to trace usage
  LLVM_DEBUG(llvm::dbgs() << "Checking if kernel arg is a scalar used inside region\n");
  
  for (Operation *kernelUser : kernelArg.getUsers()) {
    // Skip if this is the linalg.generic itself (already handled above)
    if (isa<GenericOp>(kernelUser)) continue;
    
    LLVM_DEBUG(llvm::dbgs() << "Kernel arg used by operation: " << *kernelUser << "\n");
    
    // Find the corresponding operation in actual IR using the fixed mapping
    // Note: operationMapping is actualOp -> kernelOp, so we need to reverse-search
    auto it = std::find_if(operationMapping.begin(), operationMapping.end(),
                           [kernelUser](const auto& pair) {
                             return pair.second == kernelUser;
                           });
    if (it != operationMapping.end()) {
      Operation *actualUser = it->first;  // The actual IR operation
      LLVM_DEBUG(llvm::dbgs() << "Found corresponding actual operation: " << *actualUser << "\n");
      
      // Find which operand position kernelArg occupies in kernelUser
      size_t operandIndex = 0;
      for (Value operand : kernelUser->getOperands()) {
        if (operand == kernelArg) {
          LLVM_DEBUG(llvm::dbgs() << "Kernel arg is at operand index " << operandIndex << "\n");
          
          // Get the corresponding operand from actual IR
          if (operandIndex < actualUser->getNumOperands()) {
            Value actualOperand = actualUser->getOperand(operandIndex);
            LLVM_DEBUG(llvm::dbgs() << "Found corresponding actual operand: " << actualOperand << "\n");
            return actualOperand;
          } else {
            LLVM_DEBUG(llvm::dbgs() << "ERROR - operand index out of bounds\n");
          }
          break;
        }
        operandIndex++;
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Could not find corresponding operation in operationMapping\n");
    }
  }
  
  // Fallback: if operation mapping fails, try type matching as last resort
  LLVM_DEBUG(llvm::dbgs() << "Fallback to type matching for function arguments\n");
  
  auto func = genericOp->getParentOfType<func::FuncOp>();
  if (func) {
    LLVM_DEBUG(llvm::dbgs() << "Found parent function with " << func.getNumArguments() << " arguments\n");
    
    // Look for function arguments with matching type
    for (auto funcArg : func.getArguments()) {
      if (funcArg.getType() == kernelArg.getType()) {
        LLVM_DEBUG(llvm::dbgs() << "Found function argument with matching type: " << funcArg << "\n");
        // TODO: This is still not ideal - should be improved with better analysis
        return funcArg;
      }
    }
  }
  
  LLVM_DEBUG(llvm::dbgs() << "ERROR - Could not find corresponding value for kernel arg\n");
  return nullptr;
}

// Structure to hold the result of matching a generic operation with a kernel definition
struct KernelMatchResult {
  StringRef kernelName;
  DenseMap<Operation*, Operation*> operationMapping; // actual op -> kernel op
  kernel::DefnOp matchedDefnOp;
};

// Check if a linalg.generic operation matches a kernel.defn in a collection
FailureOr<KernelMatchResult> matchGenericWithDefn(
    GenericOp genericOp, 
    kernel::DefnCollectionOp collectionOp) {
  
  // Get attributes from the generic operation
  ArrayAttr indexingMaps = genericOp.getIndexingMapsAttr();
  ArrayAttr iteratorTypes = genericOp.getIteratorTypesAttr();
  unsigned numInputs = genericOp.getNumDpsInputs();
  unsigned numOutputs = genericOp.getNumDpsInits();
  
  // Variables to capture the match result
  StringRef matchedOpName;
  DenseMap<Operation*, Operation*> matchedOperationMapping;
  kernel::DefnOp matchedDefnOp;
  
  SmallVector<kernel::DefnOp> defnOps;

      //llvm::errs() << "DEBUG: kernel.defn_collection contents:\n";
      //llvm::errs() << collectionOp;
      //llvm::errs() << collectionOp.getOperation();
      //llvm::errs() << "\n";
  collectionOp.walk([&](kernel::DefnOp defnOp) {
      defnOps.push_back(defnOp);
  });

  bool foundMatch = false;
  
  // Walk through each defn in the collection
  for (auto defnOp : defnOps) {
    
    StringRef opName = defnOp.getSymName();
    LLVM_DEBUG(llvm::dbgs() << "Checking kernel defn: " << opName << "\n");
    
    // Check for linalg.generic in the defn's body
    GenericOp candidateOp;

    defnOp.walk([&](GenericOp genericOp) {
        candidateOp = genericOp; //TODO: Add checks to make sure there is only single linalg.generic in the defn
    });

    if(!candidateOp) {
      LLVM_DEBUG(llvm::dbgs() << "No linalg.generic found in defn " << opName << "\n");
      continue;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Found linalg.generic in defn " << opName << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Candidate numInputs=" << candidateOp.getNumDpsInputs() 
                            << ", target numInputs=" << numInputs << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Candidate numOutputs=" << candidateOp.getNumDpsInits() 
                            << ", target numOutputs=" << numOutputs << "\n");
    
    // Check if this linalg.generic matches our target
    DenseMap<OpNode*, OpNode*> nodeMapping;
    DenseMap<Operation*, Operation*> operationMapping; // Added for findCorrespondingValue
    if (candidateOp.getNumDpsInputs() == numInputs &&
        candidateOp.getNumDpsInits() == numOutputs &&
        areGenericOperandTypesCompatible(genericOp, candidateOp) &&
        areIndexingMapsEquivalent(candidateOp.getIndexingMapsAttr(), indexingMaps) &&
        areIteratorTypesEquivalent(candidateOp.getIteratorTypesAttr(), iteratorTypes) &&
        areRegionsEquivalent(genericOp.getRegion(), candidateOp.getRegion(), nodeMapping, operationMapping)) {
      LLVM_DEBUG(llvm::dbgs() << "MATCH FOUND for defn " << opName << "\n");
      foundMatch = true;
      matchedOpName = opName;
      matchedOperationMapping = operationMapping; // Store the operation mapping
      matchedDefnOp = defnOp; // Store the matched defnOp
    } else {
      LLVM_DEBUG(llvm::dbgs() << "No match for defn " << opName << "\n");
      LLVM_DEBUG(llvm::dbgs() << "Input/output check: " 
                              << (candidateOp.getNumDpsInputs() == numInputs) << "\n");
      LLVM_DEBUG(llvm::dbgs() << "Maps check: " 
                              << areIndexingMapsEquivalent(candidateOp.getIndexingMapsAttr(), indexingMaps) << "\n");
      LLVM_DEBUG(llvm::dbgs() << "Iterator types check: " 
                              << areIteratorTypesEquivalent(candidateOp.getIteratorTypesAttr(), iteratorTypes) << "\n");
      LLVM_DEBUG(llvm::dbgs() << "Regions check: " 
                              << areRegionsEquivalent(genericOp.getRegion(), candidateOp.getRegion(), nodeMapping, operationMapping) << "\n");
    }
    
    if (foundMatch) {
      return KernelMatchResult{matchedOpName, matchedOperationMapping, matchedDefnOp};
    }
  }

  return failure();
}

// Rewrite pattern to convert linalg.generic to kernel ops
class LinalgGenericToKernelPattern : public OpRewritePattern<GenericOp> {
public:
  LinalgGenericToKernelPattern(MLIRContext *context, 
                              kernel::DefnCollectionOp collectionOp)
      : OpRewritePattern<GenericOp>(context), collectionOp(collectionOp) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    
    LLVM_DEBUG(llvm::dbgs() << "matchAndRewrite called for genericOp:\n");
    LLVM_DEBUG(llvm::dbgs() << genericOp << "\n");
    
    auto module = genericOp->getParentOfType<ModuleOp>();
    //Check if the parent of the generic op is a kernel.defn
    if (auto parentOp = genericOp->getParentOp()) {
      if (isa<kernel::DefnOp>(parentOp)) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping genericOp inside kernel.defn\n");
        return failure();
      }
    }
    
    // Try to match with a defn in the collection
    auto matchResult = matchGenericWithDefn(genericOp, collectionOp);
    if (failed(matchResult)) {
      LLVM_DEBUG(llvm::dbgs() << "No match found in collection\n");
      return failure();
    }
    
    StringRef opName = matchResult->kernelName;
    LLVM_DEBUG(llvm::dbgs() << "Match found with kernel: " << opName << "\n");
    
    // Find the matched kernel.defn operation
    kernel::DefnOp matchedDefnOp = matchResult->matchedDefnOp;
    
    if (!matchedDefnOp) {
      return failure();
    }
    
    // Check if the kernel.defn already exists in the target module
    kernel::DefnOp existingDefn;
    module.walk([&](kernel::DefnOp defnOp) {
      if (defnOp.getSymName() == opName) {
        // Check if this defn is inside a defn_collection (template) or at module level (callable)
        if (!defnOp->getParentOfType<kernel::DefnCollectionOp>()) {
          existingDefn = defnOp;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    
    // If the kernel.defn doesn't exist in the module, copy it
    if (!existingDefn) {
      // Clone the matched kernel.defn operation
      rewriter.setInsertionPointToStart(module.getBody());
      auto clonedDefn = rewriter.clone(*matchedDefnOp.getOperation());
      (void)clonedDefn; // Suppress unused variable warning
    }
    
    // Create kernel.launch operation to replace the genericOp
    Location loc = genericOp.getLoc();
    
    // Set insertion point to the genericOp location
    rewriter.setInsertionPoint(genericOp);
    
    // Get the kernel function signature to map all arguments
    Block &kernelBlock = matchedDefnOp.getRegion().front();
    auto kernelArgs = kernelBlock.getArguments();
    
    // Use the operationMapping from the match result (no need to call areRegionsEquivalent again)
    const DenseMap<Operation*, Operation*> &operationMapping = matchResult->operationMapping;
    
    // Use unified approach: map ALL kernel arguments to their corresponding actual values
    SmallVector<Value> operands;
    LLVM_DEBUG(llvm::dbgs() << "Starting to map " << kernelArgs.size() << " kernel arguments\n");
    
    for (BlockArgument kernelArg : kernelArgs) {
      Value actualValue = findCorrespondingValue(kernelArg, operationMapping, genericOp);
      if (!actualValue) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to find corresponding value for kernel arg #" 
                     << kernelArg.getArgNumber() << " - returning failure\n");
        return failure(); // Could not find corresponding value
      }
      operands.push_back(actualValue);
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully mapped all kernel arguments, creating kernel.launch\n");
    
    // Get kernel function signature types for casting
    auto kernelFuncType = matchedDefnOp.getFunctionType();
    auto kernelInputTypes = kernelFuncType.getInputs();
    auto kernelResultTypes = kernelFuncType.getResults();
    
    // Cast operands to match kernel signature types if needed
    SmallVector<Value> castedOperands;
    for (size_t i = 0; i < operands.size(); ++i) {
      Value operand = operands[i];
      Type expectedType = (i < kernelInputTypes.size()) ? kernelInputTypes[i] : operand.getType();
      
      if (operand.getType() != expectedType) {
        // Insert tensor.cast for type conversion
        if (isa<RankedTensorType>(operand.getType()) && isa<RankedTensorType>(expectedType)) {
          LLVM_DEBUG(llvm::dbgs() << "Casting operand " << i << " from " << operand.getType() 
                       << " to " << expectedType << "\n");
          auto castOp = rewriter.create<tensor::CastOp>(loc, expectedType, operand);
          castedOperands.push_back(castOp.getResult());
        } else {
          // For non-tensor types, use the operand as-is
          castedOperands.push_back(operand);
        }
      } else {
        castedOperands.push_back(operand);
      }
    }
    
    // Get result types from the generic operation
    TypeRange originalResultTypes = genericOp.getResultTypes();
    
    // Create the kernel.launch operation with casted operands and kernel result types
    auto launchOp = rewriter.create<kernel::LaunchOp>(
        loc, 
        kernelResultTypes,  // Use kernel result types for the launch op
        opName,
        castedOperands      // Use casted operands
    );

    // A matched linalg.generic is destination-style, but a library definition
    // may yield a computed SSA value instead of yielding its output block
    // argument directly.  Preserve the original DPS destination mapping on
    // the launch so One-Shot Bufferize can reuse the caller's output buffer
    // before the launch is lowered to a runtime call.  Operand casts do not
    // change operand numbering, so derive the mapping from the uncast values.
    SmallVector<int64_t> resultDestinations;
    resultDestinations.reserve(genericOp.getNumDpsInits());
    for (Value init : genericOp.getDpsInits()) {
      auto destination = llvm::find(operands, init);
      if (destination == operands.end()) {
        resultDestinations.clear();
        break;
      }
      resultDestinations.push_back(
          static_cast<int64_t>(std::distance(operands.begin(), destination)));
    }
    if (resultDestinations.size() == launchOp.getNumResults())
      launchOp->setAttr("polygeist.result_destinations",
                        rewriter.getDenseI64ArrayAttr(resultDestinations));
    
    // Cast results back to original types if needed
    SmallVector<Value> finalResults;
    for (size_t i = 0; i < launchOp.getResults().size(); ++i) {
      Value result = launchOp.getResult(i);
      Type originalType = (i < originalResultTypes.size()) ? originalResultTypes[i] : result.getType();
      
      if (result.getType() != originalType) {
        // Insert tensor.cast to convert back to original type
        if (isa<RankedTensorType>(result.getType()) && isa<RankedTensorType>(originalType)) {
          LLVM_DEBUG(llvm::dbgs() << "Casting result " << i << " from " << result.getType() 
                       << " to " << originalType << "\n");
          auto castOp = rewriter.create<tensor::CastOp>(loc, originalType, result);
          finalResults.push_back(castOp.getResult());
        } else {
          finalResults.push_back(result);
        }
      } else {
        finalResults.push_back(result);
      }
    }
    
    // Replace the generic operation with the final results
    rewriter.replaceOp(genericOp, finalResults);
    
    return success();
  }

private:
  kernel::DefnCollectionOp collectionOp;
};

// Pass to apply the rewrite pattern
struct LinalgToKernelPass : public LinalgToKernelBase<LinalgToKernelPass> {
  using LinalgToKernelBase::LinalgToKernelBase;
  
  // Constructor that allows setting the kernel library path
  LinalgToKernelPass() = default;
  LinalgToKernelPass(const std::string& libraryPath) : externalLibraryPath(libraryPath) {}
  
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    kernel::DefnCollectionOp collectionOp = nullptr;
    OwningOpRef<ModuleOp> externalModule;
    // Determine which path to use for kernel library
    std::string effectiveLibraryPath = externalLibraryPath;
    // If no external path was provided via constructor, try the command line option
    if (effectiveLibraryPath.empty()) {
      effectiveLibraryPath = std::string(kernelLibraryPath);
    }
    
    //// Debug output
    //llvm::errs() << "DEBUG: externalLibraryPath = '" << externalLibraryPath << "'\n";
    //llvm::errs() << "DEBUG: kernelLibraryPath = '" << std::string(kernelLibraryPath) << "'\n";
    //llvm::errs() << "DEBUG: effectiveLibraryPath = '" << effectiveLibraryPath << "'\n";
    
    // Check if we should load kernel definitions from an external file
    if (!effectiveLibraryPath.empty()) {
      //llvm::errs() << "DEBUG: Loading kernel definitions from external file: " << effectiveLibraryPath << "\n";
      // Load kernel definitions from external file
      std::string errorMessage;
      auto memoryBuffer = mlir::openInputFile(effectiveLibraryPath, &errorMessage);
      if (!memoryBuffer) {
        module.emitError("Failed to open kernel library file: ") << effectiveLibraryPath 
                         << " - " << errorMessage;
        return signalPassFailure();
      }
      
      // Parse the external file
      llvm::SourceMgr sourceMgr;
      sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
      
      externalModule = mlir::parseSourceFile<ModuleOp>(sourceMgr, &getContext());
      if (!externalModule) {
        module.emitError("Failed to parse kernel library file: ") << effectiveLibraryPath;
        return signalPassFailure();
      }
      
      // Debug: Print the loaded external module
      //llvm::errs() << "DEBUG: Successfully loaded external module:\n";
      //externalModule->print(llvm::errs());
      //llvm::errs() << "\n";
      
      // Find the kernel.defn_collection in the external module
      externalModule->walk([&](kernel::DefnCollectionOp op) {
        collectionOp = op;
        LLVM_DEBUG(llvm::dbgs() << "Found kernel.defn_collection in external module\n");
        return WalkResult::interrupt();
      });
      
      if (!collectionOp) {
        module.emitError("No kernel.defn_collection found in external kernel library: ") 
                         << effectiveLibraryPath;
        return signalPassFailure();
      }
      
      // Debug: Print the found collection
      //llvm::errs() << "DEBUG: kernel.defn_collection contents:\n";
      //llvm::errs() << collectionOp;
      //llvm::errs() << collectionOp.getOperation();
      //llvm::errs() << "\n";
    } else {
      // Find the kernel.defn_collection in the current module (original behavior)
      module.walk([&](kernel::DefnCollectionOp op) {
        collectionOp = op;
        return WalkResult::interrupt();
      });
      
      if (!collectionOp) {
        module.emitError("No kernel.defn_collection found in module. "
                         "Either include one in the input module or specify "
                         "--kernel-library-path to load from external file.");
        return signalPassFailure();
      }
    }
    
    // Apply the rewrite pattern
    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgGenericToKernelPattern>(&getContext(), collectionOp);
      
      //llvm::errs() << "DEBUG: kernel.defn_collection contents:\n";
      //llvm::errs() << collectionOp.getOperation();
      //llvm::errs() << "\n";
      //llvm::errs() << collectionOp;
      //llvm::errs() << "\n";
    
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      return signalPassFailure();
  }

private:
  std::string externalLibraryPath;
};

} // namespace

namespace mlir::polygeist {

// Create a pass to convert linalg.generic to kernel
std::unique_ptr<Pass> createLinalgToKernelPass() {
  return std::make_unique<LinalgToKernelPass>();
}

// Create a pass to convert linalg.generic to kernel with kernel library path
std::unique_ptr<Pass> createLinalgToKernelPass(const std::string& kernelLibraryPath) {
  return std::make_unique<LinalgToKernelPass>(kernelLibraryPath);
}

} // namespace mlir::polygeist
