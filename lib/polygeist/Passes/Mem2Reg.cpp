//===- Mem2Reg.cpp - MemRef DataFlow Optimization pass ------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to forward memref stores to loads, thereby
// potentially getting rid of intermediate memref's entirely.
// TODO: In the future, similar techniques could be used to eliminate
// dead memref store's and perform more complex forwarding when support for
// SSA scalars live out of 'affine.for'/'affine.if' statements is available.
//===----------------------------------------------------------------------===//
#include "PassDetails.h"
#include <iostream>
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/Passes.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Support/LLVM.h"
#include <algorithm>
#include <deque>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <set>

#include "polygeist/Ops.h"

#define DEBUG_TYPE "mem2reg"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;

typedef std::set<std::vector<ssize_t>> StoreMap;

namespace {
// The store to load forwarding relies on three conditions:
//
// 1) they need to have mathematically equivalent affine access functions
// (checked after full composition of load/store operands); this implies that
// they access the same single memref element for all iterations of the common
// surrounding loop,
//
// 2) the store op should dominate the load op,
//
// 3) among all op's that satisfy both (1) and (2), the one that postdominates
// all store op's that have a dependence into the load, is provably the last
// writer to the particular memref location being loaded at the load op, and its
// store value can be forwarded to the load. Note that the only dependences
// that are to be considered are those that are satisfied at the block* of the
// innermost common surrounding loop of the <store, load> being considered.
//
// (* A dependence being satisfied at a block: a dependence that is satisfied by
// virtue of the destination operation appearing textually / lexically after
// the source operation within the body of a 'affine.for' operation; thus, a
// dependence is always either satisfied by a loop or by a block).
//
// The above conditions are simple to check, sufficient, and powerful for most
// cases in practice - they are sufficient, but not necessary --- since they
// don't reason about loops that are guaranteed to execute at least once or
// multiple sources to forward from.
//
// TODO: more forwarding can be done when support for
// loop/conditional live-out SSA values is available.
// TODO: do general dead store elimination for memref's. This pass
// currently only eliminates the stores only if no other loads/uses (other
// than dealloc) remain.
//
struct Mem2Reg : public Mem2RegBase<Mem2Reg> {
  void runOnFunction() override;

  // return if changed
  bool forwardStoreToLoad(mlir::Value AI, std::vector<ssize_t> idx,
                          SmallVectorImpl<Operation *> &loadOpsToErase);
};

} // end anonymous namespace

/// Creates a pass to perform optimizations relying on memref dataflow such as
/// store to load forwarding, elimination of dead stores, and dead allocs.
std::unique_ptr<OperationPass<FuncOp>> mlir::polygeist::createMem2RegPass() {
  return std::make_unique<Mem2Reg>();
}

bool matchesIndices(mlir::OperandRange ops, const std::vector<ssize_t> &idx) {
  if (ops.size() != idx.size())
    return false;
  for (size_t i = 0; i < idx.size(); i++) {
    if (auto op = ops[i].getDefiningOp<ConstantIntOp>()) {
      if (op.value() != idx[i]) {
        return false;
      }
    } else if (auto op = ops[i].getDefiningOp<ConstantIndexOp>()) {
      if (op.value() != idx[i]) {
        return false;
      }
    } else {
      assert(0 && "unhandled op");
    }
  }
  return true;
}

bool matchesIndices(ArrayRef<AffineExpr> ops, const std::vector<ssize_t> &idx) {
  if (ops.size() != idx.size())
    return false;
  for (size_t i = 0; i < idx.size(); i++) {
    if (auto op = ops[i].dyn_cast<AffineConstantExpr>()) {
      if (op.getValue() != idx[i])
        return false;
    } else {
      assert(0 && "unhandled op");
    }
  }
  return true;
}

class ValueOrPlaceholder;

static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                            ValueOrPlaceholder& PH);

using BlockMap = std::map<Block*, std::shared_ptr<ValueOrPlaceholder>>;

class ValueOrPlaceholder {
	Type elType;
  	std::map<Block*, BlockArgument>& blocksWithAddedArgs;
  	BlockMap &valueAtEndOfBlock;
  	BlockMap &valueAtStartOfBlock;
  public:
    bool overwritten;
	Value val;
	Block *valueAtStart;
	scf::ExecuteRegionOp exOp;
	scf::IfOp ifOp;
	std::shared_ptr<ValueOrPlaceholder> ifLastValue;
	ValueOrPlaceholder(ValueOrPlaceholder&&) = delete;
	ValueOrPlaceholder(const ValueOrPlaceholder&) = delete;
	/*
	ValueOrPlaceholder& operator=(std::nullptr_t) {
		this->overwritten = true;
		this->val = nullptr;
		this->valueAtStart = nullptr;
		this->exOp = nullptr;
		this->ifOp = nullptr;
		this->ifLastValue = nullptr;
		return *this;
	}
	ValueOrPlaceholder& operator=(Value v) {
		assert(v);
		this->overwritten = false;
		this->val = v;
		this->valueAtStart = nullptr;
		this->exOp = nullptr;
		this->ifOp = nullptr;
		this->ifLastValue = nullptr;
		return *this;
	}
	*/
	/*
	ValueOrPlaceholder& operator=(const ValueOrPlaceholder& rhs) {
		this->overwritten = rhs.overwritten;
		this->val = rhs.val;
		this->valueAtStart = rhs.valueAtStart;
		this->exOp = rhs.exOp;
		this->ifOp = rhs.ifOp;
		this->ifLastValue = rhs.ifLastValue;
		return *this;
	}
	*/
	ValueOrPlaceholder(nullptr_t, Type elType, std::map<Block*, BlockArgument>& blocksWithAddedArgs, BlockMap &valueAtEndOfBlock, BlockMap &valueAtStartOfBlock) : 
		elType(elType),
		blocksWithAddedArgs(blocksWithAddedArgs),
		valueAtEndOfBlock(valueAtEndOfBlock),
		valueAtStartOfBlock(valueAtStartOfBlock),
		overwritten(true), val(nullptr), valueAtStart(nullptr), exOp(nullptr), ifOp(nullptr), ifLastValue() {
	}
	ValueOrPlaceholder(Value val, Type elType, std::map<Block*, BlockArgument>& blocksWithAddedArgs, BlockMap &valueAtEndOfBlock, BlockMap &valueAtStartOfBlock) : 
		elType(elType),
		blocksWithAddedArgs(blocksWithAddedArgs),
		valueAtEndOfBlock(valueAtEndOfBlock),
		valueAtStartOfBlock(valueAtStartOfBlock),
		overwritten(false), val(val), valueAtStart(nullptr), exOp(nullptr), ifOp(nullptr), ifLastValue() {
		assert(val);
	}
	ValueOrPlaceholder(Block* valueAtStart, Type elType, std::map<Block*, BlockArgument>& blocksWithAddedArgs, BlockMap &valueAtEndOfBlock, BlockMap &valueAtStartOfBlock) : 
		elType(elType),
		blocksWithAddedArgs(blocksWithAddedArgs),
		valueAtEndOfBlock(valueAtEndOfBlock),
		valueAtStartOfBlock(valueAtStartOfBlock),
		overwritten(false), val(nullptr), valueAtStart(valueAtStart), exOp(nullptr), ifOp(nullptr), ifLastValue() {
		assert(valueAtStart);
	}
	ValueOrPlaceholder(scf::ExecuteRegionOp exOp, Type elType, std::map<Block*, BlockArgument>& blocksWithAddedArgs, BlockMap &valueAtEndOfBlock, BlockMap &valueAtStartOfBlock) : 
		elType(elType),
		blocksWithAddedArgs(blocksWithAddedArgs),
		valueAtEndOfBlock(valueAtEndOfBlock),
		valueAtStartOfBlock(valueAtStartOfBlock),
		overwritten(false), val(nullptr), valueAtStart(nullptr), exOp(exOp), ifOp(nullptr), ifLastValue() {
		assert(exOp);
	}
	ValueOrPlaceholder(scf::IfOp ifOp, std::shared_ptr<ValueOrPlaceholder> ifLastVal, Type elType, std::map<Block*, BlockArgument>& blocksWithAddedArgs, BlockMap &valueAtEndOfBlock, BlockMap&valueAtStartOfBlock) : 
		elType(elType),
		blocksWithAddedArgs(blocksWithAddedArgs),
		valueAtEndOfBlock(valueAtEndOfBlock),
		valueAtStartOfBlock(valueAtStartOfBlock),
		overwritten(false), val(nullptr), valueAtStart(nullptr), exOp(nullptr), ifOp(ifOp), ifLastValue(ifLastVal) {
		assert(ifOp);
	}
    // Return true if this represents a full expression if all block argsare defined at start
    // Append the list of blocks requiring definition to block.
    bool definedWithArg(SmallPtrSetImpl<Block*>& block) {
		if (val) return true;
		if (overwritten) return false;
		if (valueAtStart) {
			auto found = valueAtStartOfBlock.find(valueAtStart);
			if (found != valueAtStartOfBlock.end()) {
                if (found->second->valueAtStart != valueAtStart)
                    return found->second->definedWithArg(block);
			}
            block.insert(valueAtStart);
            return true;
		}
		if (ifOp) {
			auto thenFind = valueAtEndOfBlock.find(ifOp.thenBlock());
			assert(thenFind != valueAtEndOfBlock.end());
            assert(thenFind->second);
			if (!thenFind->second->definedWithArg(block)) return false;
			
			if (ifOp.getElseRegion().getBlocks().size()) {
				auto elseFind = valueAtEndOfBlock.find(ifOp.thenBlock());
				assert(elseFind != valueAtEndOfBlock.end());
                assert(elseFind->second);
				if (!elseFind->second->definedWithArg(block)) return false;
			} else {
				if (!ifLastValue->definedWithArg(block)) return false;
			}
			return true;
		}
		if (exOp) {
			for (auto &B : exOp.getRegion()) {
				if (auto yield = dyn_cast<scf::YieldOp>(B.getTerminator())) {
					auto found = valueAtEndOfBlock.find(&B);
					assert(found != valueAtEndOfBlock.end());
                    assert(found->second);
					if (!found->second->definedWithArg(block)) return false;
				}
			}
			return true;
		}
		assert(0 && "unhandled");
	}
	Value materialize(bool full=true) {
		if (overwritten) return nullptr;
		if (val) return val;
		if (valueAtStart) {
			auto found = valueAtStartOfBlock.find(valueAtStart);
			if (found != valueAtStartOfBlock.end()) {
				if (found->second->valueAtStart != valueAtStart)
					return found->second->materialize(full);
				//valueAtStart = nullptr;
				//return this->val = found->second;
			}
			//auto found2 = blocksWithAddedArgs.find(valueAtStart);
			//if (found2 != blocksWithAddedArgs.end())
			//	return found2->second;//->materialize();
			if (!full) return nullptr;
			llvm::errs() << " could not get valueAtStart: " << valueAtStart << "; ";
			if (found != valueAtStartOfBlock.end()) {
                llvm::errs() << " map vas: " << *found->second << "\n";
            } else {
                llvm::errs() << " no map\n";
            }
			Block* blk = valueAtStart;
			blk->dump();
    		assert(0 && "no null");
		}
		if (exOp) return materializeEx(full);
		if (ifOp) return materializeIf(full);
		assert(0 && "");
	}

	Value materializeEx(bool full=true) {
		assert(exOp);
                
		SmallVector<scf::YieldOp> yields;
		SmallVector<Value>        values;
		std::set<size_t> equivalent;
		for(size_t i=0, num=exOp.getNumResults(); i<num; i++)
			equivalent.insert(i);
		for (auto &B : exOp.getRegion()) {
			if (auto yield = dyn_cast<scf::YieldOp>(B.getTerminator())) {
				auto found = valueAtEndOfBlock.find(&B);
				assert(found != valueAtEndOfBlock.end());
                assert(found->second);
				Value post = found->second->materialize(full);
				if (found->second->overwritten) {
					this->overwritten = true;
					this->exOp = nullptr;
					return nullptr;
				}
				if (!post) {
					if (full) {
						this->overwritten = true;
						this->exOp = nullptr;
					}
					return nullptr;
				}
				yields.push_back(yield);
				values.push_back(post);
				for(auto pair : llvm::enumerate(yield.getOperands()))
					if (pair.value() != post)
						equivalent.erase(pair.index());
			}
		}

		// Must contain only region invariant results.
		bool allSame = true;
		for (auto v : values)
			allSame &= v == values[0];

		// If all all paths are the same, and the value is not defined within the 
		// execute region, simply return that single value, rather than creating a new return.
		if (allSame) {
			if (values[0].getDefiningOp() && !exOp->isAncestor(values[0].getDefiningOp())) {
				this->exOp = nullptr;
				return this->val = values[0];
			}
			if (auto ba = values[0].dyn_cast<BlockArgument>())
				if (!exOp->isAncestor(ba.getOwner()->getParentOp())) {
					this->exOp = nullptr;
					return this->val = values[0];
				}
		}
		// If there's an equivalent return, don't create a new return and instead use that result.
		if (equivalent.size() > 0) {
			this->val = exOp.getResult(*equivalent.begin());
			this->exOp = nullptr;
			return this->val;
		}

		OpBuilder B(exOp.getContext());
		B.setInsertionPoint(exOp);
		SmallVector<mlir::Type, 4> tys(exOp.getResultTypes().begin(),
									   exOp.getResultTypes().end());
		tys.push_back(elType);
		auto nextEx = B.create<mlir::scf::ExecuteRegionOp>(
			exOp.getLoc(), tys);

		nextEx.getRegion().takeBody(exOp.getRegion());
		for (auto pair : llvm::zip(yields, values)) {
		  SmallVector<Value, 4> vals = std::get<0>(pair).getOperands();
		  vals.push_back(std::get<1>(pair));
		  std::get<0>(pair)->setOperands(vals);
		}
		
		SmallVector<mlir::Value, 3> resvals = nextEx.getResults();
		this->val = resvals.back();
		resvals.pop_back();
		exOp.replaceAllUsesWith(resvals);
		//StoringOperations.erase(exOp);
		//StoringOperations.insert(nextEx);
		exOp.erase();
		this->exOp = nullptr;
		return this->val;
	}
	Value materializeIf(bool full=true) {
		auto thenFind = valueAtEndOfBlock.find(ifOp.thenBlock());
		assert(thenFind != valueAtEndOfBlock.end());
		assert(thenFind->second);
		Value thenVal = thenFind->second->materialize(full);
		if (thenFind->second->overwritten) {
				this->overwritten = true;
				this->ifLastValue = nullptr;
				this->ifOp = nullptr;
				return nullptr;
		}
		if (!thenVal) {
			if (full) {
				this->overwritten = true;
				this->ifLastValue = nullptr;
				this->ifOp = nullptr;
			}
			return nullptr;
		}
        Value elseVal;

        if (ifOp.getElseRegion().getBlocks().size()) {
			auto elseFind = valueAtEndOfBlock.find(ifOp.thenBlock());
			assert(elseFind != valueAtEndOfBlock.end());
            assert(elseFind->second);
			elseVal = elseFind->second->materialize(full);
			if (elseFind->second->overwritten) {
				this->overwritten = true;
				this->ifLastValue = nullptr;
				this->ifOp = nullptr;
				return nullptr;
			}
        } else {
 			elseVal = ifLastValue->materialize(full);
			if (ifLastValue->overwritten) {
				this->overwritten = true;
				this->ifLastValue = nullptr;
				this->ifOp = nullptr;
				return nullptr;
			}
		}
		
		if (!elseVal) {
			if (full) {
				this->overwritten = true;
				this->ifLastValue = nullptr;
				this->ifOp = nullptr;
			}
			return nullptr;
		}
              
              if (thenVal == elseVal) {
				this->overwritten = false;
				this->ifLastValue = nullptr;
				this->ifOp = nullptr;
                return this->val = thenVal;
              }

                if (ifOp.getElseRegion().getBlocks().size()) {
                  for (auto tup : llvm::reverse(llvm::zip(
                           ifOp.getResults(), ifOp.thenYield().getOperands(),
                           ifOp.elseYield().getOperands()))) {
                    if (std::get<1>(tup) == thenVal &&
                        std::get<2>(tup) == elseVal) {
				this->overwritten = false;
				this->ifLastValue = nullptr;
				this->ifOp = nullptr;
                return this->val = thenVal;
                    }
                  }
                }

                OpBuilder B(ifOp.getContext());
                B.setInsertionPoint(ifOp);
                SmallVector<mlir::Type, 4> tys(ifOp.getResultTypes().begin(),
                                               ifOp.getResultTypes().end());
                tys.push_back(thenVal.getType());
                auto nextIf = B.create<mlir::scf::IfOp>(
                    ifOp.getLoc(), tys, ifOp.getCondition(), /*hasElse*/ true);

                SmallVector<mlir::Value, 4> thenVals = ifOp.thenYield().getResults();
                thenVals.push_back(thenVal);
                nextIf.getThenRegion().takeBody(ifOp.getThenRegion());
				nextIf.thenYield()->setOperands(thenVals);

                if (ifOp.getElseRegion().getBlocks().size()) {
                  nextIf.getElseRegion().getBlocks().clear();
                  SmallVector<mlir::Value, 4> elseVals =
					ifOp.elseYield().getResults();
                  elseVals.push_back(elseVal);
                  nextIf.getElseRegion().takeBody(ifOp.getElseRegion());
                  nextIf.elseYield()->setOperands(elseVals);
                } else {
                  B.setInsertionPoint(&nextIf.getElseRegion().back(),
                                      nextIf.getElseRegion().back().begin());
                  SmallVector<mlir::Value, 4> elseVals = {elseVal};
                  B.create<mlir::scf::YieldOp>(ifOp.getLoc(), elseVals);
                }

                SmallVector<mlir::Value, 3> resvals = nextIf.getResults();
                this->val = resvals.back();
                resvals.pop_back();
                ifOp.replaceAllUsesWith(resvals);

                //StoringOperations.erase(ifOp);
                //StoringOperations.insert(nextIf);
                ifOp.erase();
				this->overwritten = false;
				this->ifLastValue = nullptr;
				this->ifOp = nullptr;
                return this->val;
	}
};
static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                            ValueOrPlaceholder& PH) {
  if (PH.overwritten)
	return os << "<overwritten>";
  if (PH.val)
	return os << PH.val;
  if (PH.valueAtStart) {
	PH.valueAtStart->print(os);
	return os;;
	}
  if (PH.ifOp)
	return os << PH.ifOp << " -<lastVal:" << *PH.ifLastValue << ">";
  if (PH.exOp)
	return os << PH.exOp;
  return os;
}


struct Analyzer {

  const std::set<Block *> &Good;
  const std::set<Block *> &Bad;
  const std::set<Block *> &Other;
  std::set<Block *> Legal;
  std::set<Block *> Illegal;
  size_t depth;
  Analyzer(const std::set<Block *> &Good, const std::set<Block *> &Bad,
           const std::set<Block *> &Other, std::set<Block *> Legal,
           std::set<Block *> Illegal, size_t depth = 0)
      : Good(Good), Bad(Bad), Other(Other), Legal(Legal), Illegal(Illegal),
        depth(depth) {}

  void analyze() {
    while (1) {
      std::deque<Block *> todo(Other.begin(), Other.end());
      todo.insert(todo.end(), Good.begin(), Good.end());
      while (todo.size()) {
        auto block = todo.front();
        todo.pop_front();
        if (Legal.count(block) || Illegal.count(block))
          continue;
        bool currentlyLegal = !block->hasNoPredecessors();
        for (auto pred : block->getPredecessors()) {
          if (Bad.count(pred)) {
            assert(!Legal.count(block));
            Illegal.insert(block);
            currentlyLegal = false;
            for (auto succ : block->getSuccessors()) {
              todo.push_back(succ);
            }
            break;
          } else if (Good.count(pred) || Legal.count(pred)) {
            continue;
          } else if (Illegal.count(pred)) {
            Illegal.insert(block);
            currentlyLegal = false;
            for (auto succ : block->getSuccessors()) {
              todo.push_back(succ);
            }
            break;
          } else {
            /*
            if (!Other.count(pred)) {
              pred->getParentOp()->dump();
              pred->dump();
              llvm::errs() << " - pred ptr: " << pred << "\n";
            }
            assert(Other.count(pred));
            */
            currentlyLegal = false;
            break;
          }
        }
        if (currentlyLegal) {
          Legal.insert(block);
          assert(!Illegal.count(block));
          for (auto succ : block->getSuccessors()) {
            todo.push_back(succ);
          }
        }
      }
      bool changed = false;
      for (auto O : Other) {
        if (Legal.count(O) || Illegal.count(O))
          continue;
        Analyzer AssumeLegal(Good, Bad, Other, Legal, Illegal, depth + 1);
        AssumeLegal.Legal.insert(O);
        AssumeLegal.analyze();
        bool currentlyLegal = true;
        for (auto pred : O->getPredecessors()) {
          if (!AssumeLegal.Legal.count(pred) && !AssumeLegal.Good.count(pred)) {
            currentlyLegal = false;
            break;
          }
        }
        if (currentlyLegal) {
          Legal.insert(AssumeLegal.Legal.begin(), AssumeLegal.Legal.end());
          assert(!Illegal.count(O));
          changed = true;
          break;
        } else {
          Illegal.insert(O);
          assert(!Legal.count(O));
        }
      }
      if (!changed)
        break;
    }
  }
};
			  /*
			  Operation* newLoad = nullptr;
              if (lastVal == nullptr && lastValIsStartOfBlock) {
                OpBuilder B(exOp.getContext());
                B.setInsertionPoint(exOp);
                SmallVector<mlir::Value, 4> nidx;
                for (auto i : idx) {
                  nidx.push_back(B.create<ConstantIndexOp>(exOp.getLoc(), i));
                }
                if (AI.getType().isa<MemRefType>()) {
				  SmallVector<AffineExpr> exprs;
				  for (auto i : idx)
					exprs.push_back(B.getAffineConstantExpr(i));
                  AffineMap m = AffineMap::get(0, 0, exprs, B.getContext());
				  newLoad = B.create<AffineLoadOp>(exOp.getLoc(), AI, m, ValueRange());
                } else
                  newLoad = B.create<LLVM::LoadOp>(exOp.getLoc(), AI);
			    lastVal = newLoad->getResult(0);
				lastValIsStartOfBlock = false;
				newLoads.insert(newLoad);
              }
			  */

// Remove block arguments if possible
void removeRedundantBlockArgs(Value AI, Type elType, std::map<Block*, BlockArgument> &blocksWithAddedArgs) {
    std::deque<Block *> todo;
	for(auto & p : blocksWithAddedArgs)
		todo.push_back(p.first);
    while (todo.size()) {
      auto block = todo.front();
      todo.pop_front();
      if (!blocksWithAddedArgs.count(block))
        continue;

      BlockArgument blockArg = blocksWithAddedArgs.find(block)->second;
      if (blockArg.getOwner() != block) continue;
      assert(blockArg.getOwner() == block);

      mlir::Value val = nullptr;
      bool legal = true;

      SetVector<Block *> prepred(block->getPredecessors().begin(),
                                 block->getPredecessors().end());
      for (auto pred : prepred) {
        mlir::Value pval = nullptr;

        if (auto op = dyn_cast<BranchOp>(pred->getTerminator())) {
          pval = op.getOperands()[blockArg.getArgNumber()];
          if (pval.getType() != elType) {
            pval.getDefiningOp()->getParentRegion()->getParentOp()->dump();
            llvm::errs() << pval << " - " << AI << "\n";
          }
          assert(pval.getType() == elType);
          if (pval == blockArg)
            pval = nullptr;
        } else if (auto op = dyn_cast<CondBranchOp>(pred->getTerminator())) {
          if (op.getTrueDest() == block) {
            if (blockArg.getArgNumber() >= op.getTrueOperands().size()) {
              block->dump();
              llvm::errs() << op << " ba: " << blockArg.getArgNumber() << "\n";
            }
            assert(blockArg.getArgNumber() < op.getTrueOperands().size());
            pval = op.getTrueOperands()[blockArg.getArgNumber()];
            assert(pval.getType() == elType);
            if (pval == blockArg)
              pval = nullptr;
          }
          if (op.getFalseDest() == block) {
            assert(blockArg.getArgNumber() < op.getFalseOperands().size());
            auto pval2 = op.getFalseOperands()[blockArg.getArgNumber()];
            assert(pval2.getType() == elType);
            if (pval2 != blockArg) {
              if (pval == nullptr) {
                pval = pval2;
              } else if (pval != pval2) {
                legal = false;
                break;
              }
            }
            if (pval == blockArg)
              pval = nullptr;
          }
        } else if (auto op = dyn_cast<SwitchOp>(pred->getTerminator())) {
          mlir::OpBuilder subbuilder(op.getOperation());
          if (op.getDefaultDestination() == block) {
            pval = op.getDefaultOperands()[blockArg.getArgNumber()];
            if (pval == blockArg)
              pval = nullptr;
          }
          for (auto pair : llvm::enumerate(op.getCaseDestinations())) {
            if (pair.value() == block) {
              auto pval2 =
                  op.getCaseOperands(pair.index())[blockArg.getArgNumber()];
              if (pval2 != blockArg) {
                if (pval == nullptr)
                  pval = pval2;
                else if (pval != pval2) {
                  legal = false;
                  break;
                }
              }
            }
          }
          if (legal == false)
            break;
        } else {
          llvm::errs() << *pred->getParent()->getParentOp() << "\n";
          pred->dump();
          block->dump();
          assert(0 && "unknown branch");
        }

        assert(pval != blockArg);
        if (val == nullptr) {
          val = pval;
          if (pval)
            assert(val.getType() == elType);
        } else {
          if (pval != nullptr && val != pval) {
            legal = false;
            break;
          }
        }
      }
      if (legal)
        assert(val || block->hasNoPredecessors());

      bool used = false;
      for (auto U : blockArg.getUsers()) {

        if (auto op = dyn_cast<BranchOp>(U)) {
          size_t i = 0;
          for (auto V : op.getOperands()) {
            if (V == blockArg &&
                !(i == blockArg.getArgNumber() && op.getDest() == block)) {
              used = true;
              break;
            }
          }
          if (used)
            break;
        } else if (auto op = dyn_cast<CondBranchOp>(U)) {
          size_t i = 0;
          for (auto V : op.getTrueOperands()) {
            if (V == blockArg &&
                !(i == blockArg.getArgNumber() && op.getTrueDest() == block)) {
              used = true;
              break;
            }
          }
          if (used)
            break;
          i = 0;
          for (auto V : op.getFalseOperands()) {
            if (V == blockArg &&
                !(i == blockArg.getArgNumber() && op.getFalseDest() == block)) {
              used = true;
              break;
            }
          }
        } else
          used = true;
      }
      if (!used) {
        legal = true;
      }

      if (legal) {
        for (auto U : blockArg.getUsers()) {
          if (auto block = U->getBlock()) {
            todo.push_back(block);
            for (auto succ : block->getSuccessors())
              todo.push_back(succ);
          }
        }
        if (val != nullptr) {
          if (blockArg.getType() != val.getType()) {
            block->dump();
            llvm::errs() << " AI: " << AI << "\n";
            llvm::errs() << blockArg << " val " << val << "\n";
          }
          assert(blockArg.getType() == val.getType());
          blockArg.replaceAllUsesWith(val);
        } else {
        }

        SetVector<Block *> prepred(block->getPredecessors().begin(),
                                   block->getPredecessors().end());
        for (auto pred : prepred) {
          if (auto op = dyn_cast<BranchOp>(pred->getTerminator())) {
            mlir::OpBuilder subbuilder(op.getOperation());
            std::vector<Value> args(op.getOperands().begin(),
                                    op.getOperands().end());
            args.erase(args.begin() + blockArg.getArgNumber());
            assert(args.size() == op.getOperands().size() - 1);
            subbuilder.create<BranchOp>(op.getLoc(), op.getDest(), args);
            op.erase();
          } else if (auto op = dyn_cast<CondBranchOp>(pred->getTerminator())) {

            mlir::OpBuilder subbuilder(op.getOperation());
            std::vector<Value> trueargs(op.getTrueOperands().begin(),
                                        op.getTrueOperands().end());
            std::vector<Value> falseargs(op.getFalseOperands().begin(),
                                         op.getFalseOperands().end());
            if (op.getTrueDest() == block) {
              trueargs.erase(trueargs.begin() + blockArg.getArgNumber());
            }
            if (op.getFalseDest() == block) {
              falseargs.erase(falseargs.begin() + blockArg.getArgNumber());
            }
            assert(trueargs.size() < op.getTrueOperands().size() ||
                   falseargs.size() < op.getFalseOperands().size());
            subbuilder.create<CondBranchOp>(op.getLoc(), op.getCondition(),
                                            op.getTrueDest(), trueargs,
                                            op.getFalseDest(), falseargs);
            op.erase();
          } else if (auto op = dyn_cast<SwitchOp>(pred->getTerminator())) {
            mlir::OpBuilder builder(op.getOperation());
            SmallVector<Value> defaultOps(op.getDefaultOperands().begin(),
                                          op.getDefaultOperands().end());
            if (op.getDefaultDestination() == block)
              defaultOps.erase(defaultOps.begin() + blockArg.getArgNumber());

            SmallVector<SmallVector<Value>> cases;
            SmallVector<ValueRange> vrange;
            for (auto pair : llvm::enumerate(op.getCaseDestinations())) {
              cases.emplace_back(op.getCaseOperands(pair.index()));
              if (pair.value() == block) {
                cases.back().erase(cases.back().begin() +
                                   blockArg.getArgNumber());
              }
              vrange.push_back(cases.back());
            }
            builder.create<mlir::SwitchOp>(op.getLoc(), op.getFlag(),
                                           op.getDefaultDestination(),
                                           defaultOps, op.getCaseValuesAttr(),
                                           op.getCaseDestinations(), vrange);
            op.erase();
          }
        }
        block->eraseArgument(blockArg.getArgNumber());
        blocksWithAddedArgs.erase(block);
      }
    }
  }
// This is a straightforward implementation not optimized for speed. Optimize
// if needed.
bool Mem2Reg::forwardStoreToLoad(mlir::Value AI, std::vector<ssize_t> idx,
                                 SmallVectorImpl<Operation *> &loadOpsToErase) {
  bool changed = false;
  std::set<mlir::Operation *> loadOps;
  mlir::Type subType = nullptr;
  std::set<mlir::Operation *> allStoreOps;

  std::deque<std::pair<mlir::Value, /*indexed*/ bool>> list = {{AI, false}};

  SmallPtrSet<Operation *, 4> AliasingStoreOperations;

  LLVM_DEBUG(llvm::dbgs() << "Begin forwarding store of " << AI << " to load\n"
                          << *AI.getDefiningOp()->getParentOfType<FuncOp>()
                          << "\n");
  bool captured = false;
  while (list.size()) {
    auto pair = list.front();
    auto val = pair.first;
    auto modified = pair.second;
    list.pop_front();
    for (auto *user : val.getUsers()) {
      if (auto co = dyn_cast<mlir::memref::CastOp>(user)) {
        list.emplace_back((Value)co, false);
        continue;
      }
      if (auto co = dyn_cast<polygeist::Memref2PointerOp>(user)) {
        list.emplace_back((Value)co, false);
        continue;
      }
      if (auto co = dyn_cast<polygeist::Pointer2MemrefOp>(user)) {
        list.emplace_back((Value)co, false);
        continue;
      }
      if (auto co = dyn_cast<polygeist::SubIndexOp>(user)) {
        list.emplace_back((Value)co, true);
        continue;
      }
      if (auto co = dyn_cast<mlir::LLVM::GEPOp>(user)) {
        list.emplace_back((Value)co, true);
        continue;
      }
      if (auto co = dyn_cast<mlir::LLVM::BitcastOp>(user)) {
        list.emplace_back((Value)co, false);
        continue;
      }
      if (auto loadOp = dyn_cast<mlir::memref::LoadOp>(user)) {
        if (matchesIndices(loadOp.getIndices(), idx)) {
          subType = loadOp.getType();
          loadOps.insert(loadOp);
          LLVM_DEBUG(llvm::dbgs() << "Matching Load: " << loadOp << "\n");
          continue;
        }
      }
      if (!modified) {
        if (auto loadOp = dyn_cast<mlir::LLVM::LoadOp>(user)) {
          subType = loadOp.getType();
          loadOps.insert(loadOp);
          LLVM_DEBUG(llvm::dbgs() << "Matching Load: " << loadOp << "\n");
          continue;
        }
        if (auto loadOp = dyn_cast<AffineLoadOp>(user)) {
          if (matchesIndices(loadOp.getAffineMapAttr().getValue().getResults(),
                             idx)) {
            subType = loadOp.getType();
            loadOps.insert(loadOp);
            LLVM_DEBUG(llvm::dbgs() << "Matching Load: " << loadOp << "\n");
            continue;
          }
        }
        if (auto storeOp = dyn_cast<mlir::memref::StoreOp>(user)) {
          if (storeOp.getMemRef() == val &&
              matchesIndices(storeOp.getIndices(), idx)) {
            LLVM_DEBUG(llvm::dbgs() << "Matching Store: " << storeOp << "\n");
            allStoreOps.insert(storeOp);
            continue;
          }
        }
        if (auto storeOp = dyn_cast<LLVM::StoreOp>(user)) {
          if (storeOp.getAddr() == val) {
            LLVM_DEBUG(llvm::dbgs() << "Matching Store: " << storeOp << "\n");
            allStoreOps.insert(storeOp);
            continue;
          } else {
            captured = true;
          }
        }
        if (auto storeOp = dyn_cast<AffineStoreOp>(user)) {
          if (storeOp.getMemRef() == val) {
            if (matchesIndices(
                    storeOp.getAffineMapAttr().getValue().getResults(), idx)) {
              LLVM_DEBUG(llvm::dbgs() << "Matching Store: " << storeOp << "\n");
              allStoreOps.insert(storeOp);
              continue;
            }
          } else {
            captured = true;
          }
        }
      }
      if (auto callOp = dyn_cast<mlir::CallOp>(user)) {
        if (callOp.getCallee() != "free") {
          LLVM_DEBUG(llvm::dbgs() << "Aliasing Store: " << callOp << "\n");
          AliasingStoreOperations.insert(callOp);
          captured = true;
        }
      }
      if (auto callOp = dyn_cast<mlir::LLVM::CallOp>(user)) {
        if (*callOp.getCallee() != "free") {
          LLVM_DEBUG(llvm::dbgs() << "Aliasing Store: " << callOp << "\n");
          AliasingStoreOperations.insert(callOp);
          captured = true;
        }
      }
    }
  }

  if (captured) {
    AI.getDefiningOp()->getParentOp()->walk([&](Operation *op) {
      bool opMayHaveEffect = false;
      if (op->hasTrait<OpTrait::HasRecursiveSideEffects>())
        return;
      MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
      if (!interface)
        opMayHaveEffect = true;
      if (interface) {
        SmallVector<MemoryEffects::EffectInstance, 1> effects;
        interface.getEffects(effects);

        for (auto effect : effects) {
          // If op causes EffectType on a potentially aliasing location for
          // memOp, mark as having the effect.
          if (isa<MemoryEffects::Write>(effect.getEffect())) {
            if (effect.getValue() &&
                (effect.getValue().getDefiningOp<memref::AllocaOp>() ||
                 effect.getValue().getDefiningOp<memref::AllocOp>() ||
                 effect.getValue().getDefiningOp<LLVM::AllocaOp>())) {
              if (effect.getValue() != AI)
                continue;
            }
            opMayHaveEffect = true;
            break;
          }
        }
      }
      if (opMayHaveEffect) {
        LLVM_DEBUG(llvm::dbgs() << "Potential Op ith Effect: " << *op << "\n");
        AliasingStoreOperations.insert(op);
      }
    });
  }

  if (loadOps.size() == 0) {
    return changed;
  }
  /*
  // this is a valid optimization, however it should occur naturally
  // from the logic to follow anyways
  if (allStoreOps.size() == 1) {
    auto store = *allStoreOps.begin();
    for(auto loadOp : loadOps) {
      if (domInfo->dominates(store, loadOp)) {
        loadOp.replaceAllUsesWith(store.getValueToStore());
        loadOpsToErase.push_back(loadOp);
      }
    }
    return changed;
  }
  */
  assert(AI.getDefiningOp());
  Region *parentAI = AI.getDefiningOp()->getParentRegion();
  assert(parentAI);

  // A list of all regions which contain loads to be replaced.
  SmallPtrSet<Region *, 4> ContainsLoadingOperation;
  {
    SmallVector<Region *> todo;
    for (auto load : loadOps) {
        todo.push_back(load->getParentRegion());
    }
    while (todo.size()) {
        auto op = todo.back();
        todo.pop_back();
        if (ContainsLoadingOperation.contains(op)) continue;
        if (op == parentAI) continue;
        ContainsLoadingOperation.insert(op);
        auto parent = op->getParentRegion();
        assert(parent);
        todo.push_back(parent);
    }
  }

  // List of operations which may store that are not storeops
  SmallPtrSet<Operation *, 4> StoringOperations;
  SmallPtrSet<Block *, 4> StoringBlocks;
  {
    std::deque<Block *> todo;
    for (auto &pair : allStoreOps) {
      LLVM_DEBUG(llvm::dbgs() << " storing operation: " << *pair << "\n");
      todo.push_back(pair->getBlock());
    }
    for (auto op : AliasingStoreOperations) {
      StoringOperations.insert(op);
      LLVM_DEBUG(llvm::dbgs()
                 << " aliasing storing operation: " << *op << "\n");
      todo.push_back(op->getBlock());
    }
    while (todo.size()) {
      auto block = todo.front();
      assert(block);
      todo.pop_front();
      StoringBlocks.insert(block);
      LLVM_DEBUG(llvm::dbgs() << " initial storing block: " << block << "\n");
      if (auto op = block->getParentOp()) {
        StoringOperations.insert(op);
        if (auto next = op->getBlock()) {
          StoringBlocks.insert(next);
          LLVM_DEBUG(llvm::dbgs()
                     << " derived storing block: " << next << "\n");
          todo.push_back(next);
        }
      }
    }
  }

  Type elType;
  if (auto MT = AI.getType().dyn_cast<MemRefType>())
    elType = MT.getElementType();
  else
    elType = AI.getType().cast<LLVM::LLVMPointerType>().getElementType();

  // Last value stored in an individual block and the operation which stored it
  BlockMap valueAtEndOfBlock;

  // Last value stored in an individual block and the operation which stored it
  BlockMap valueAtStartOfBlock;

  std::map<Block*, BlockArgument> blocksWithAddedArgs;

  SmallVector<std::pair<Value, std::shared_ptr<ValueOrPlaceholder>>> replaceableValues;

  DenseMap<Value, std::shared_ptr<ValueOrPlaceholder>> metaMap;
  auto getValue = [&](Value orig) {
	assert(orig);
	auto found = metaMap.find(orig);
	if (found != metaMap.end()) return found->second;
	return metaMap.try_emplace(orig, std::make_shared<ValueOrPlaceholder>(orig, elType, blocksWithAddedArgs, valueAtEndOfBlock, valueAtStartOfBlock)).first->second;
  };
  auto emptyValue = std::make_shared<ValueOrPlaceholder>(nullptr, elType, blocksWithAddedArgs, valueAtEndOfBlock, valueAtStartOfBlock); 

  auto replaceValue = [&](Value orig, std::shared_ptr<ValueOrPlaceholder> replacement) -> std::shared_ptr<ValueOrPlaceholder> {
    assert(replacement);
	replacement->materialize(/*full*/false);
    assert(orig.getType() == elType);
	if (replacement->overwritten) {
        loadOps.erase(orig.getDefiningOp()); 
		return getValue(orig);
	} else if (replacement->val) {
	  changed = true;
	  assert(orig != replacement->val);
      assert(replacement->val.getType() == elType);
      assert(orig.getType() == replacement->val.getType() &&
             "mismatched load type");
      LLVM_DEBUG(llvm::dbgs() << " replaced " << orig << " with "
                              << replacement->val << "\n");
      orig.replaceAllUsesWith(replacement->val);

	  // This doesn't replace all things.
	  if (metaMap.find(orig) != metaMap.end()) {
		metaMap.find(orig)->second->val = replacement->val;
		metaMap.erase(orig);
      } else {
	  }
      // Record this to erase later.
      loadOpsToErase.push_back(orig.getDefiningOp());
      loadOps.erase(orig.getDefiningOp());
      return replacement;
    } else {
      assert(replacement);
	  replaceableValues.push_back(std::pair<Value,std::shared_ptr<ValueOrPlaceholder>>(orig, replacement));
      assert(replaceableValues.back().second);
      return getValue(orig);
    }
  };

  // Start by setting valueAtEndOfBlock to the last store directly in that block
  // Note that this may miss a store within a region of an operation in that
  // block
  // endRequires denotes whether this value is needed at the end of the block
  // (yield)
  std::function<void(Block &, std::shared_ptr<ValueOrPlaceholder>)>
      handleBlock = [&](Block &block, std::shared_ptr<ValueOrPlaceholder> lastVal) {
        valueAtStartOfBlock.emplace(&block, lastVal);
        SmallVector<Operation *, 10> ops;
        for (auto &a : block) {
          ops.push_back(&a);
        }
        LLVM_DEBUG(llvm::dbgs() << "\nstarting block: lastVal="<< *lastVal << "\n";
                   block.print(llvm::dbgs()); llvm::dbgs() << "\n";
                   );
        for (auto a : ops) {
          if (StoringOperations.count(a)) {
            if (auto exOp = dyn_cast<mlir::scf::ExecuteRegionOp>(a)) {
			  for (auto &B : exOp.getRegion())
				handleBlock(B, (&B == &exOp.getRegion().front()) ? lastVal : std::make_shared<ValueOrPlaceholder>(&B, elType, blocksWithAddedArgs, valueAtEndOfBlock, valueAtStartOfBlock));
			  lastVal = std::make_shared<ValueOrPlaceholder>(exOp, elType, blocksWithAddedArgs, valueAtEndOfBlock, valueAtStartOfBlock);
              continue;
            } else if (auto ifOp = dyn_cast<mlir::scf::IfOp>(a)) {
			  handleBlock(*ifOp.getThenRegion().begin(), lastVal);
			  if (ifOp.getElseRegion().getBlocks().size()) {
                handleBlock(*ifOp.getElseRegion().begin(), lastVal);
              }
			  lastVal = std::make_shared<ValueOrPlaceholder>(ifOp, lastVal, elType, blocksWithAddedArgs, valueAtEndOfBlock, valueAtStartOfBlock);
              continue;
            }
            LLVM_DEBUG(llvm::dbgs() << "erased store due to: " << *a << "\n");
            
            for (auto &R : a->getRegions())
                if (ContainsLoadingOperation.contains(&R)) {
                    for (auto &B : R) {
                        handleBlock(B, (&B == &R.front()) ? emptyValue :
														    std::make_shared<ValueOrPlaceholder>(&B, elType, blocksWithAddedArgs, valueAtEndOfBlock, valueAtStartOfBlock));
                    }
                }
            lastVal = emptyValue;
          } else if (loadOps.count(a)) {
            Value loadOp = a->getResult(0);
            lastVal = replaceValue(loadOp, lastVal);
          } else if (auto storeOp = dyn_cast<memref::StoreOp>(a)) {
            if (allStoreOps.count(storeOp)) {
              lastVal = getValue(storeOp.getValueToStore());
            }
          } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(a)) {
            if (allStoreOps.count(storeOp)) {
              lastVal = getValue(storeOp.getValue());
            }
          } else if (auto storeOp = dyn_cast<AffineStoreOp>(a)) {
            if (allStoreOps.count(storeOp)) {
              lastVal = getValue(storeOp.getValueToStore());
            }
          } else {
            // since not storing operation the value at the start and end of
            // block is lastVal
            a->walk([&](Operation *op) {
              if (loadOps.count(op)) {
                replaceValue(op->getResult(0), lastVal);
              }
            });
          }
        }
        LLVM_DEBUG(llvm::dbgs() << " ending block: "; block.print(llvm::dbgs());
                   llvm::dbgs() << " with val:" << *lastVal << "\n";);
        assert(lastVal);
        valueAtEndOfBlock.emplace(&block, lastVal);
      };

  {
	assert(AI.getDefiningOp());
	SmallVector<Block*> todo = {AI.getDefiningOp()->getBlock()};
	SmallPtrSet<Block*, 2> done;
	while (todo.size()) {
		Block* cur = todo.back();
		todo.pop_back();
		if (done.contains(cur)) continue;
		done.insert(cur);
		if (cur == AI.getDefiningOp()->getBlock())
	        handleBlock(*cur, emptyValue);
		else
        	handleBlock(*cur, std::make_shared<ValueOrPlaceholder>(cur, elType, blocksWithAddedArgs, valueAtEndOfBlock, valueAtStartOfBlock));
		for (auto B : cur->getSuccessors())
			todo.push_back(B);
	}
  }

  if (loadOps.size() == 0)
    return changed;

  std::set<Block *> Good;
  std::set<Block *> Bad;
  std::set<Block *> Other;

  for (auto &pair : valueAtEndOfBlock) {
    assert(pair.second);
    SmallPtrSet<Block*, 1> requirements;
    if (pair.second->definedWithArg(requirements)) {
      if (requirements.size() == 0)
        Good.insert(pair.first);
      else
        Other.insert(pair.first);
      // llvm::errs() << "<GOOD: " << " - " << AI << " " << pair.first << ">\n";
      // pair.first->dump();
      // llvm::errs() << "</GOOD: " << " - " << AI << ">\n";
    } else if (pair.second->overwritten) {//StoringBlocks.count(pair.first)) {
      // llvm::errs() << "<BAD: " << " - " << AI << " " << pair.first << ">\n";
      // pair.first->dump();
      // llvm::errs() << "</BAD: " << " - " << AI << ">\n";
      Bad.insert(pair.first);
    }
  }

  {
    std::deque<Block *> todo;
    for (auto B : Good)
      for (auto succ : B->getSuccessors())
        todo.push_back(succ);
    while (todo.size()) {
      auto block = todo.front();
      todo.pop_front();
      if (Good.count(block) || Bad.count(block) || Other.count(block))
        continue;
      if (StoringBlocks.count(block)) {
        // llvm::errs() << "<BAD2: " << " - " << AI << " " << block << ">\n";
        // block->dump();
        // llvm::errs() << "</BAD2: " << " - " << AI << ">\n";
        Bad.insert(block);
        continue;
      }
      Other.insert(block);
      // llvm::errs() << "<OTHER2: " << " - " << AI << " " << block << ">\n";
      // block->dump();
      // llvm::errs() << "</OTHER2: " << " - " << AI << ">\n";
      if (isa<BranchOp, CondBranchOp, SwitchOp>(block->getTerminator())) {
        for (auto succ : block->getSuccessors()) {
          todo.push_back(succ);
        }
      }
    }
  }

  Analyzer A(Good, Bad, Other, {}, {});
  A.analyze();

  for (auto block : A.Legal) {
     //llvm::errs() << "<LEGAL: " << " - " << AI << " " << block << ">\n";
     //block->dump();
	 //llvm::errs() << " startofblock: " << (int)(valueAtStartOfBlock.find(block) == valueAtStartOfBlock.end() ) << "\n";
	 //if (valueAtStartOfBlock.find(block) != valueAtStartOfBlock.end() ) llvm::errs() << " found: " << * valueAtStartOfBlock.find(block)->second << "\n";
     //llvm::errs() << "</LEGAL: " << " - " << AI << ">\n";
    if (valueAtStartOfBlock.find(block) == valueAtStartOfBlock.end() || valueAtStartOfBlock.find(block)->second->valueAtStart != block)
      continue;
     //llvm::errs() << "<SLEGAL: " << " - " << AI << " " << block << ">\n";
     //block->dump();
     //llvm::errs() << "</SLEGAL: " << " - " << AI << ">\n";
    auto arg = block->addArgument(subType);
	std::shared_ptr<ValueOrPlaceholder> argVal = getValue(arg);
    valueAtStartOfBlock[block] = argVal;
    blocksWithAddedArgs[block] = arg;
    /*
	for (Operation &op : *block) {
      if (!StoringOperations.count(&op)) {
        op.walk([&](Block *blk) {
          if (valueAtStartOfBlock.find(blk) == valueAtStartOfBlock.end()) {
            valueAtStartOfBlock[blk] = arg;
            if (valueAtEndOfBlock.find(blk) == valueAtEndOfBlock.end() ||
                StoringBlocks.count(blk) == 0) {
              valueAtEndOfBlock[blk] = arg;
            }
          }
        });
      } else
        break;
    }
	*/
    if (valueAtEndOfBlock.find(block) == valueAtEndOfBlock.end() ||
        StoringBlocks.count(block) == 0) {
      assert(argVal);
      valueAtEndOfBlock.insert(std::make_pair(block, argVal));
    }
  }
  for (auto block : A.Illegal) {
      valueAtStartOfBlock[block] = emptyValue;
  }

  for (auto& pair : replaceableValues) {
    assert(pair.first);
    assert(pair.second);
	Value val = pair.second->materialize(false);
    if (!val) {
        loadOps.erase(pair.first.getDefiningOp()); 
        continue;
    }
	assert(val);
	  
        changed = true;
	  assert(pair.first != val);
      assert(val.getType() == elType);
      assert(pair.first.getType() == val.getType() &&
             "mismatched load type");
      LLVM_DEBUG(llvm::dbgs() << " replaced " << pair.first << " with "
                              << val << "\n");
      pair.first.replaceAllUsesWith(val);

	  // This doesn't replace all things.
	  if (metaMap.find(pair.first) != metaMap.end()) {
		metaMap.find(pair.first)->second->val = val;
		metaMap.erase(pair.first);
      } else {
	  }
      // Record this to erase later.
      loadOpsToErase.push_back(pair.first.getDefiningOp());
      loadOps.erase(pair.first.getDefiningOp());
  }
  replaceableValues.clear();

  for (auto &pair : blocksWithAddedArgs) {
	Block* block = pair.first;
    assert(valueAtStartOfBlock.find(block) != valueAtStartOfBlock.end());

    Value maybeblockArg = valueAtStartOfBlock.find(block)->second->materialize(false);
    auto blockArg = maybeblockArg.dyn_cast<BlockArgument>();
    assert(blockArg && blockArg.getOwner() == block);

    SetVector<Block *> prepred(block->getPredecessors().begin(),
                               block->getPredecessors().end());
    for (auto pred : prepred) {
      assert(pred && "Null predecessor");
	  assert(valueAtEndOfBlock.find(pred) != valueAtEndOfBlock.end());
      assert(valueAtEndOfBlock.find(pred)->second);
      mlir::Value pval = valueAtEndOfBlock.find(pred)->second->materialize(true);
      if (!pval) {
        AI.getDefiningOp()->getParentOfType<FuncOp>().dump();
        pred->dump();
        llvm::errs() << "pval: " << *valueAtEndOfBlock.find(pred)->second << "\n";
      }
      assert(pval && "Null last stored");
      assert(pred->getTerminator());

      assert(blockArg.getOwner() == block);
      if (auto op = dyn_cast<BranchOp>(pred->getTerminator())) {
        mlir::OpBuilder subbuilder(op.getOperation());
        std::vector<Value> args(op.getOperands().begin(),
                                op.getOperands().end());
        args.push_back(pval);
        subbuilder.create<BranchOp>(op.getLoc(), op.getDest(), args);
        // op.replaceAllUsesWith(op2);
        op.erase();
      } else if (auto op = dyn_cast<CondBranchOp>(pred->getTerminator())) {

        mlir::OpBuilder subbuilder(op.getOperation());
        std::vector<Value> trueargs(op.getTrueOperands().begin(),
                                    op.getTrueOperands().end());
        std::vector<Value> falseargs(op.getFalseOperands().begin(),
                                     op.getFalseOperands().end());
        if (op.getTrueDest() == block) {
          trueargs.push_back(pval);
        }
        if (op.getFalseDest() == block) {
          falseargs.push_back(pval);
        }
        subbuilder.create<CondBranchOp>(op.getLoc(), op.getCondition(),
                                        op.getTrueDest(), trueargs,
                                        op.getFalseDest(), falseargs);
        // op.replaceAllUsesWith(op2);
        op.erase();
      } else if (auto op = dyn_cast<SwitchOp>(pred->getTerminator())) {
        mlir::OpBuilder builder(op.getOperation());
        SmallVector<Value> defaultOps(op.getDefaultOperands().begin(),
                                      op.getDefaultOperands().end());

        if (op.getDefaultDestination() == block)
          defaultOps.push_back(pval);

        SmallVector<SmallVector<Value>> cases;
        SmallVector<ValueRange> vrange;
        for (auto pair : llvm::enumerate(op.getCaseDestinations())) {
          cases.emplace_back(op.getCaseOperands(pair.index()).begin(),
                             op.getCaseOperands(pair.index()).end());
          if (pair.value() == block) {
            cases.back().push_back(pval);
          }
          vrange.push_back(cases.back());
        }
        builder.create<mlir::SwitchOp>(
            op.getLoc(), op.getFlag(), op.getDefaultDestination(), defaultOps,
            op.getCaseValuesAttr(), op.getCaseDestinations(), vrange);
        op.erase();
      } else {
        llvm_unreachable("unknown pred branch");
      }
    }
  }

  removeRedundantBlockArgs(AI, elType, blocksWithAddedArgs);

  for (auto loadOp : llvm::make_early_inc_range(loadOps)) {
    assert(loadOp);
	if (loadOp->getResult(0).use_empty()) {
      loadOpsToErase.push_back(loadOp);
      loadOps.erase(loadOp);
    }
  }
  return changed;
}

bool isPromotable(mlir::Value AI) {
  std::deque<mlir::Value> list = {AI};

  while (list.size()) {
    auto val = list.front();
    list.pop_front();

    for (auto U : val.getUsers()) {
      if (auto LO = dyn_cast<memref::LoadOp>(U)) {
        for (auto idx : LO.getIndices()) {
          if (!idx.getDefiningOp<ConstantIntOp>() &&
              !idx.getDefiningOp<ConstantIndexOp>()) {
            // llvm::errs() << "non promotable "; AI.dump(); llvm::errs() << "
            // ldue to " << idx << "\n";
            return false;
          }
        }
        continue;
      } else if (auto LO = dyn_cast<LLVM::LoadOp>(U)) {
        continue;
      } else if (auto SO = dyn_cast<LLVM::StoreOp>(U)) {
        continue;
      } else if (auto LO = dyn_cast<AffineLoadOp>(U)) {
        for (auto idx : LO.getAffineMapAttr().getValue().getResults()) {
          if (!idx.isa<AffineConstantExpr>()) {
            return false;
          }
        }
        continue;
      } else if (auto SO = dyn_cast<memref::StoreOp>(U)) {
        if (SO.value() == val)
          return false;
        for (auto idx : SO.getIndices()) {
          if (!idx.getDefiningOp<ConstantIntOp>() &&
              !idx.getDefiningOp<ConstantIndexOp>()) {
            // llvm::errs() << "non promotable "; AI.dump(); llvm::errs() << "
            // sdue to " << idx << "\n";
            return false;
          }
        }
        continue;
      } else if (auto SO = dyn_cast<AffineStoreOp>(U)) {
        if (SO.value() == val)
          return false;
        for (auto idx : SO.getAffineMapAttr().getValue().getResults()) {
          if (!idx.isa<AffineConstantExpr>()) {
            return false;
          }
        }
        continue;
      } else if (isa<memref::DeallocOp>(U)) {
        continue;
      } else if (isa<CallOp>(U) && cast<CallOp>(U).getCallee() == "free") {
        continue;
      } else if (isa<CallOp>(U)) {
        // TODO check "no capture", currently assume as a fallback always
        // nocapture
        continue;
      } else if (auto CO = dyn_cast<memref::CastOp>(U)) {
        list.push_back(CO);
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "non promotable " << AI << " due to " << *U << "\n");
        return false;
      }
    }
  }
  return true;
}

StoreMap getLastStored(mlir::Value AI) {
  StoreMap lastStored;

  std::deque<mlir::Value> list = {AI};

  while (list.size()) {
    auto val = list.front();
    list.pop_front();
    for (auto U : val.getUsers()) {
      if (auto SO = dyn_cast<memref::StoreOp>(U)) {
        std::vector<ssize_t> vec;
        for (auto idx : SO.getIndices()) {
          if (auto op = idx.getDefiningOp<ConstantIntOp>()) {
            vec.push_back(op.value());
          } else if (auto op = idx.getDefiningOp<ConstantIndexOp>()) {
            vec.push_back(op.value());
          } else {
            assert(0 && "unhandled op");
          }
        }
        lastStored.insert(vec);
      } else if (auto SO = dyn_cast<AffineLoadOp>(U)) {
        std::vector<ssize_t> vec;
        for (auto idx : SO.getAffineMapAttr().getValue().getResults()) {
          if (auto op = idx.dyn_cast<AffineConstantExpr>()) {
            vec.push_back(op.getValue());
          } else {
            assert(0 && "unhandled op");
          }
        }
        lastStored.insert(vec);
      } else if (isa<LLVM::LoadOp>(U)) {
        std::vector<ssize_t> vec;
        lastStored.insert(vec);
      } else if (isa<LLVM::StoreOp>(U)) {
        std::vector<ssize_t> vec;
        lastStored.insert(vec);
      } else if (auto SO = dyn_cast<memref::LoadOp>(U)) {
        std::vector<ssize_t> vec;
        for (auto idx : SO.getIndices()) {
          if (auto op = idx.getDefiningOp<ConstantIntOp>()) {
            vec.push_back(op.value());
          } else if (auto op = idx.getDefiningOp<ConstantIndexOp>()) {
            vec.push_back(op.value());
          } else {
            assert(0 && "unhandled op");
          }
        }
        lastStored.insert(vec);
      } else if (auto SO = dyn_cast<AffineStoreOp>(U)) {
        std::vector<ssize_t> vec;
        for (auto idx : SO.getAffineMapAttr().getValue().getResults()) {
          if (auto op = idx.dyn_cast<AffineConstantExpr>()) {
            vec.push_back(op.getValue());
          } else {
            assert(0 && "unhandled op");
          }
        }
        lastStored.insert(vec);
      } else if (auto CO = dyn_cast<memref::CastOp>(U)) {
        list.push_back(CO);
      }
    }
  }
  return lastStored;
}

void Mem2Reg::runOnFunction() {
  // Only supports single block functions at the moment.
  FuncOp f = getFunction();

  // Variable indicating that a memref has had a load removed
  // and or been deleted. Because there can be memrefs of
  // memrefs etc, we may need to do multiple passes (first
  // to eliminate the outermost one, then inner ones)
  bool changed;
  FuncOp freeRemoved = nullptr;
  do {
    changed = false;

    // A list of memref's that are potentially dead / could be eliminated.
    SmallPtrSet<Value, 4> memrefsToErase;

    // Load op's whose results were replaced by those forwarded from stores.
    SmallVector<Operation *, 8> loadOpsToErase;

    // Walk all load's and perform store to load forwarding.
    SmallVector<mlir::Value, 4> toPromote;
    f.walk([&](mlir::memref::AllocaOp AI) {
      if (isPromotable(AI)) {
        toPromote.push_back(AI);
      }
    });
    f.walk([&](mlir::memref::AllocOp AI) {
      if (isPromotable(AI)) {
        toPromote.push_back(AI);
      }
    });
    f.walk([&](LLVM::AllocaOp AI) {
      if (isPromotable(AI)) {
        toPromote.push_back(AI);
      }
    });

    for (auto AI : toPromote) {
      LLVM_DEBUG(llvm::dbgs() << " attempting to promote " << AI << "\n");
      auto lastStored = getLastStored(AI);
      for (auto &vec : lastStored) {
        LLVM_DEBUG(llvm::dbgs() << " + forwarding vec to promote {";
                   for (auto m
                        : vec) llvm::dbgs()
                   << (int)m << ",";
                   llvm::dbgs() << "} of " << AI << "\n");
        // llvm::errs() << " PRE " << AI << "\n";
        // f.dump();
        changed |= forwardStoreToLoad(AI, vec, loadOpsToErase);
        // llvm::errs() << " POST " << AI << "\n";
        // f.dump();
      }
      memrefsToErase.insert(AI);
    }

    // Erase all load op's whose results were replaced with store fwd'ed ones.
    for (auto *loadOp : loadOpsToErase) {
      changed = true;
      loadOp->erase();
    }

    // Check if the store fwd'ed memrefs are now left with only stores and can
    // thus be completely deleted. Note: the canonicalize pass should be able
    // to do this as well, but we'll do it here since we collected these anyway.
    for (auto memref : memrefsToErase) {

      // If the memref hasn't been alloc'ed in this function, skip.
      Operation *defOp = memref.getDefiningOp();
      if (!defOp ||
          !(isa<memref::AllocOp>(defOp) || isa<memref::AllocaOp>(defOp) ||
            isa<LLVM::AllocaOp>(defOp)))
        // TODO: if the memref was returned by a 'call' operation, we
        // could still erase it if the call had no side-effects.
        continue;

      std::deque<mlir::Value> list = {memref};
      std::vector<mlir::Operation *> toErase;
      bool error = false;
      while (list.size()) {
        auto val = list.front();
        list.pop_front();

        for (auto U : val.getUsers()) {
          if (auto SO = dyn_cast<LLVM::StoreOp>(U)) {
            if (SO.getValue() == val) {
              error = true;
              break;
            }
            toErase.push_back(U);
          } else if (auto SO = dyn_cast<memref::StoreOp>(U)) {
            if (SO.value() == val) {
              error = true;
              break;
            }
            toErase.push_back(U);
          } else if (auto SO = dyn_cast<AffineStoreOp>(U)) {
            if (SO.value() == val) {
              error = true;
              break;
            }
            toErase.push_back(U);
          } else if (isa<memref::DeallocOp>(U)) {
            toErase.push_back(U);
          } else if (isa<CallOp>(U) && cast<CallOp>(U).getCallee() == "free") {
            toErase.push_back(U);
          } else if (auto CO = dyn_cast<memref::CastOp>(U)) {
            toErase.push_back(U);
            list.push_back(CO);
          } else if (auto CO = dyn_cast<polygeist::SubIndexOp>(U)) {
            toErase.push_back(U);
            list.push_back(CO);
          } else {
            error = true;
            break;
          }
        }
        if (error)
          break;
      }

      if (!error) {
        std::reverse(toErase.begin(), toErase.end());
        for (auto *user : toErase) {
          user->erase();
        }
        defOp->erase();
        changed = true;
      } else {
        // llvm::errs() << " failed to remove: " << memref << "\n";
      }
    }
  } while (changed);

  if (freeRemoved) {
    if (freeRemoved.use_empty()) {
      freeRemoved.erase();
    }
  }
}
