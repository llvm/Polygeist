#include <clang/Basic/Version.h>
#include <clang/Basic/Builtins.h>
#include <clang/Basic/FileSystemOptions.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Tool.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/HeaderSearchOptions.h>
#include <clang/Basic/LangStandard.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Frontend/Utils.h>
#include <clang/Lex/HeaderSearch.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/Pragma.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/Sema/Sema.h>
#include <clang/Sema/SemaDiagnostic.h>
#include <clang/Parse/Parser.h>
#include <clang/Parse/ParseAST.h>

using namespace std;
using namespace clang;
using namespace clang::driver;
using namespace llvm::opt;
#include "clang/AST/StmtVisitor.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

struct ValueWithOffsets {
    mlir::Value val;
    std::vector<mlir::Value> offsets;
    ValueWithOffsets() = default;
    ValueWithOffsets(ValueWithOffsets &v) = default;
    ValueWithOffsets(ValueWithOffsets &&v) = default;

    ValueWithOffsets(mlir::Value val, std::vector<mlir::Value> offsets) : val(val), offsets(offsets) {}
    template<typename T>
    ValueWithOffsets(T sval) : val(sval), offsets() {}
    template<typename T>
    ValueWithOffsets(T& sval) : val(sval), offsets() {}
    operator mlir::Value() const { 
        assert(offsets.size() == 0);
        return val;
    }
    mlir::Type getType() { return val.getType(); }
};

struct MLIRScanner : public StmtVisitor<MLIRScanner, ValueWithOffsets> {
public:
	codegen::MLIRCodegen &mcg;
    mlir::Location loc;

    std::vector<std::map<std::string, mlir::Value>> scopes;

    void setValue(std::string name, mlir::Value val) {
        scopes.back()[name] = val;
    }

    mlir::Value getValue(std::string name) {
        for(int i=scopes.size()-1; i>=0; i--) {
            auto found = scopes[i].find(name);
            if (found != scopes[i].end()) {
                return found->second;
            }
        }
        llvm::errs() << "couldn't find " << name << "\n";
        assert(0 && "couldnt find value");
    }

    mlir::Type getLLVMTypeFromMLIRType(mlir::Type t) {
        if (auto it = t.dyn_cast<mlir::IntegerType>()) {
            return mlir::LLVM::LLVMIntegerType::get(t.getContext(), it.getWidth());
        }
        assert(0 && "unhandled mlir=>llvm type");
    }

    mlir::Type getMLIRType(const clang::Type* t) {
        if (t->isSpecificBuiltinType(BuiltinType::Float)) {
            return mcg.builder_.getF32Type();
        }
        if (t->isSpecificBuiltinType(BuiltinType::Void)) {
            return mcg.builder_.getNoneType();
        }
        if (t->isSpecificBuiltinType(BuiltinType::Int) || t->isSpecificBuiltinType(BuiltinType::UInt)) {
            return mcg.builder_.getI32Type();
        }
        if (auto pt = dyn_cast<clang::PointerType>(t)) {
            return mlir::MemRefType::get(-1, getMLIRType(&*pt->getPointeeType()));
        }
        if (auto pt = dyn_cast<clang::ConstantArrayType>(t)) {
            auto under = getMLIRType(&*pt->getElementType());
            if (auto mt = under.dyn_cast<mlir::MemRefType>()) {
                auto shape2 = std::vector<int64_t>(mt.getShape());
                shape2.insert(shape2.begin(), (int64_t)pt->getSize().getLimitedValue());
                return mlir::MemRefType::get(shape2, mt.getElementType(), mt.getAffineMaps(), mt.getMemorySpace());
            }
            return mlir::MemRefType::get({(int64_t)pt->getSize().getLimitedValue()}, under);
        }
        if (auto pt = dyn_cast<clang::IncompleteArrayType>(t)) {
            auto under = getMLIRType(&*pt->getElementType());
            if (auto mt = under.dyn_cast<mlir::MemRefType>()) {
                auto shape2 = std::vector<int64_t>(mt.getShape());
                shape2.insert(shape2.begin(), -1);
                return mlir::MemRefType::get(shape2, mt.getElementType(), mt.getAffineMaps(), mt.getMemorySpace());
            }
            return mlir::MemRefType::get({-1}, under);
        }
        //if (auto pt = dyn_cast<clang::RecordType>(t)) {
        //    llvm::errs() << " thing: " << pt->getName() << "\n";
        //}
        t->dump();
        assert(0 && "unknown type to convert");
        return nullptr;
    }

    mlir::Value createAllocOp(mlir::Type t, std::string name, uint64_t memspace, bool isArray=false) {
        mlir::MemRefType mr;
        if (!isArray) {
            mr = mlir::MemRefType::get(1, t, {}, memspace);
        } else {
            auto mt = t.cast<mlir::MemRefType>();
            mr = mlir::MemRefType::get(mt.getShape(), mt.getElementType(), mt.getAffineMaps(), memspace);            
        }
        auto alloc = mcg.builder_.create<mlir::AllocaOp>(loc, mr);
        setValue(name, alloc);
        return alloc;
    }

    mlir::Value createAndSetAllocOp(std::string name, mlir::Value v, uint64_t memspace) {
        auto op = createAllocOp(v.getType(), name, memspace);
        mlir::Value zeroIndex = getConstantIndex(0);
        mcg.builder_.create<mlir::StoreOp>(loc, v, op, zeroIndex);
        return op;
    }

    mlir::Block* entryBlock;
    mlir::FuncOp function;
    MLIRScanner(FunctionDecl *fd, codegen::MLIRCodegen &mcg) : mcg(mcg), loc(mcg.builder_.getUnknownLoc()) {
        llvm::errs() << *fd << "\n";
        if (mcg.theModule_.lookupSymbol(fd->getName())) {
            return;
        }
        scopes.emplace_back();
        std::vector<mlir::Type> types;
        std::vector<std::string> names;
        for(auto parm : fd->parameters()) {
            types.push_back(getMLIRType(&*parm->getOriginalType()));
            names.push_back(parm->getName().str());
        }
        
        //auto argTypes = getFunctionArgumentsTypes(mcg.getContext(), inputTensors);
        auto rt = getMLIRType(&*fd->getReturnType());
        std::vector<mlir::Type> rettypes;
        if (!rt.isa<mlir::NoneType>()) {
            rettypes.push_back(rt);
        }
        auto funcType = mcg.builder_.getFunctionType(types, rettypes);
        function = mlir::FuncOp(mlir::FuncOp::create(loc, fd->getName(), funcType));

        entryBlock = function.addEntryBlock();

        mcg.builder_.setInsertionPointToStart(entryBlock);
        mcg.theModule_.push_back(function);

        for (unsigned i = 0, e = function.getNumArguments(); i != e; ++i) {
            //function.getArgument(i).setName(names[i]);
            createAndSetAllocOp(names[i], function.getArgument(i), 0);
        }


        scopes.emplace_back();

        Stmt *stmt = fd->getBody();
        stmt->dump();
        Visit(stmt);

        auto endBlock = mcg.builder_.getInsertionBlock();
        if (endBlock->empty() || endBlock->back().isKnownNonTerminator()) {
            mcg.builder_.create<mlir::ReturnOp>(loc);
        }
        function.dump();
    }

    ValueWithOffsets VisitDeclStmt(clang::DeclStmt* decl) {
        for(auto sub : decl->decls()) {
            if (auto vd = dyn_cast<VarDecl>(sub)) {
                VisitVarDecl(vd);
            } else {
                llvm::errs() << " + visiting unknonwn sub decl stmt\n";
                sub->dump();
                assert(0 && "unknown sub decl");
            }
        }
        return nullptr;
    }

    ValueWithOffsets VisitIntegerLiteral(clang::IntegerLiteral* expr) {
        auto ty = getMLIRType(&*expr->getType()).cast<mlir::IntegerType>();
        return mcg.builder_.create<mlir::ConstantOp>(loc, ty, mcg.builder_.getIntegerAttr(ty, expr->getValue()));
    }

    ValueWithOffsets VisitParenExpr(clang::ParenExpr* expr) {
        return Visit(expr->getSubExpr());
    }

    ValueWithOffsets VisitVarDecl(clang::VarDecl* decl) {
        unsigned memtype = 0;

        //if (decl->hasAttr<CUDADeviceAttr>() || decl->hasAttr<CUDAConstantAttr>() ||
        if (decl->hasAttr<CUDASharedAttr>()) {
            memtype = 5;
        }
        auto op = createAllocOp(getMLIRType(&*decl->getType()), decl->getName().str(), memtype, /*isArray*/isa<clang::ArrayType>(decl->getType()));
        if (auto init = decl->getInit()) {
            auto inite = Visit(init);
            if (!(mlir::Value)inite) {
                init->dump();
            }
            assert((mlir::Value)inite);
            mlir::Value zeroIndex = getConstantIndex(0);
            mcg.builder_.create<mlir::StoreOp>(loc, inite, op, zeroIndex);
        }
        return ValueWithOffsets(op, {});
    }

    ValueWithOffsets VisitForStmt(clang::ForStmt* fors) {
        scopes.emplace_back();

        if (auto s = fors->getInit()) {
            Visit(s);
        }


        auto &condB = *function.addBlock();
        auto &bodyB = *function.addBlock();

        auto &exitB = *function.addBlock();

        mcg.builder_.create<mlir::BranchOp>(loc, &condB);

        mcg.builder_.setInsertionPointToStart(&condB);

        if(auto s = fors->getCond()) {
            auto condRes = Visit(s);
            mcg.builder_.create<mlir::CondBranchOp>(loc, (mlir::Value)condRes, &bodyB, &exitB);
        }

        mcg.builder_.setInsertionPointToStart(&bodyB);
        Visit(fors->getBody());
        if(auto s = fors->getInc()) {
            Visit(s);
        }
        mcg.builder_.create<mlir::BranchOp>(loc, &condB);

        mcg.builder_.setInsertionPointToStart(&exitB);
        scopes.pop_back();
        return nullptr;
    }

    ValueWithOffsets VisitArraySubscriptExpr(clang::ArraySubscriptExpr* expr) {
        auto lhs = Visit(expr->getLHS());
        auto rhs = Visit(expr->getRHS());
        auto v = lhs.offsets;
        v.push_back((mlir::Value) mcg.builder_.create<mlir::IndexCastOp>(loc, rhs, mlir::IndexType::get(rhs.val.getContext())));
        return ValueWithOffsets(lhs.val, v);
    }

    ValueWithOffsets VisitCallExpr(clang::CallExpr* expr) {

        if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
        if (auto ME = dyn_cast<MemberExpr>(ic->getSubExpr())) {
            auto memberName = ME->getMemberDecl()->getName() ;
        if (auto sr2 = dyn_cast<OpaqueValueExpr>(ME->getBase())) {
        if (auto sr = dyn_cast<DeclRefExpr>(sr2->getSourceExpr())) {
            if (sr->getDecl()->getName() == "blockIdx") {
                auto mlirType = getMLIRType(&*expr->getType());
                if (memberName == "__fetch_builtin_x") {
                    return mcg.builder_.create<mlir::IndexCastOp>(loc,
                        mcg.builder_.create<mlir::gpu::BlockIdOp>(loc, mlir::IndexType::get(mcg.builder_.getContext()), "x"), mlirType);
                }
                if (memberName == "__fetch_builtin_y") {
                    return mcg.builder_.create<mlir::IndexCastOp>(loc,
                        mcg.builder_.create<mlir::gpu::BlockIdOp>(loc, mlir::IndexType::get(mcg.builder_.getContext()), "y"), mlirType);
                }
                if (memberName == "__fetch_builtin_z") {
                    return mcg.builder_.create<mlir::IndexCastOp>(loc,
                        mcg.builder_.create<mlir::gpu::BlockIdOp>(loc, mlir::IndexType::get(mcg.builder_.getContext()), "z"), mlirType);
                }
            }
            if (sr->getDecl()->getName() == "blockDim") {
                auto mlirType = getMLIRType(&*expr->getType());
                if (memberName == "__fetch_builtin_x") {
                    return mcg.builder_.create<mlir::IndexCastOp>(loc,
                        mcg.builder_.create<mlir::gpu::BlockDimOp>(loc, mlir::IndexType::get(mcg.builder_.getContext()), "x"), mlirType);
                }
                if (memberName == "__fetch_builtin_y") {
                    return mcg.builder_.create<mlir::IndexCastOp>(loc,
                        mcg.builder_.create<mlir::gpu::BlockDimOp>(loc, mlir::IndexType::get(mcg.builder_.getContext()), "y"), mlirType);
                }
                if (memberName == "__fetch_builtin_z") {
                    return mcg.builder_.create<mlir::IndexCastOp>(loc,
                        mcg.builder_.create<mlir::gpu::BlockDimOp>(loc, mlir::IndexType::get(mcg.builder_.getContext()), "z"), mlirType);
                }
            }
            if (sr->getDecl()->getName() == "threadIdx") {
                auto mlirType = getMLIRType(&*expr->getType());
                auto llvmType = getLLVMTypeFromMLIRType(mlirType);
                if (memberName == "__fetch_builtin_x") {
                    return mcg.builder_.create<mlir::IndexCastOp>(loc,
                        mcg.builder_.create<mlir::gpu::ThreadIdOp>(loc, mlir::IndexType::get(mcg.builder_.getContext()), "x"), mlirType);
                }
                if (memberName == "__fetch_builtin_y") {
                    return mcg.builder_.create<mlir::IndexCastOp>(loc,
                        mcg.builder_.create<mlir::gpu::ThreadIdOp>(loc, mlir::IndexType::get(mcg.builder_.getContext()), "y"), mlirType);
                }
                if (memberName == "__fetch_builtin_z") {
                    return mcg.builder_.create<mlir::IndexCastOp>(loc,
                        mcg.builder_.create<mlir::gpu::ThreadIdOp>(loc, mlir::IndexType::get(mcg.builder_.getContext()), "z"), mlirType);
                }
            }
            if (sr->getDecl()->getName() == "gridDim") {
                auto mlirType = getMLIRType(&*expr->getType());
                auto llvmType = getLLVMTypeFromMLIRType(mlirType);
                if (memberName == "__fetch_builtin_x") {
                    return mcg.builder_.create<mlir::IndexCastOp>(loc,
                        mcg.builder_.create<mlir::gpu::GridDimOp>(loc, mlir::IndexType::get(mcg.builder_.getContext()), "x"), mlirType);
                }
                if (memberName == "__fetch_builtin_y") {
                    return mcg.builder_.create<mlir::IndexCastOp>(loc,
                        mcg.builder_.create<mlir::gpu::GridDimOp>(loc, mlir::IndexType::get(mcg.builder_.getContext()), "y"), mlirType);
                }
                if (memberName == "__fetch_builtin_z") {
                    return mcg.builder_.create<mlir::IndexCastOp>(loc,
                        mcg.builder_.create<mlir::gpu::GridDimOp>(loc, mlir::IndexType::get(mcg.builder_.getContext()), "z"), mlirType);
                }
            }
        }}}

        if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
        if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
            if (sr->getDecl()->getName() == "__syncthreads") {
                mcg.builder_.create<mlir::NVVM::Barrier0Op>(loc);
                return nullptr;
            }
        }
        
        if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
        if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
            if (sr->getDecl()->getName() == "__shfl_up_sync") {
                assert(0 && "__shfl_up_sync unhandled");
                //mcg.builder_.create<mlir::NVVM::ShflBflyOp>(loc);
                return nullptr;
            }
        }


        auto tocall = (mlir::Value)Visit(expr->getCallee());
        std::vector<mlir::Value> args;
        for(auto a : expr->arguments()) {
            args.push_back(Visit(a));
        }
        if (auto fo = tocall.getDefiningOp<mlir::FuncOp>()) {
            return mcg.builder_.create<mlir::CallOp>(loc, fo, args).getResult(0);
        }
        llvm::errs() << "do not support indirecto call of " << tocall << "\n";
        assert(0 && "no indirect");
    }

    std::map<int, mlir::Value> constants;
    ValueWithOffsets getConstantIndex(int x) {
        if (constants.find(x) != constants.end()) {
            return ValueWithOffsets(constants[x], {});
        }
        mlir::OpBuilder builder(mcg.builder_.getContext());
        builder.setInsertionPointToStart(entryBlock);
        return ValueWithOffsets(constants[x] = builder.create<mlir::ConstantIndexOp>(loc, x), {});
    }

    ValueWithOffsets VisitMSPropertyRefExpr(MSPropertyRefExpr* expr) {
        expr->getBaseExpr()->dump();
        expr->getPropertyDecl()->dump();
        // TODO obviously fake
        return getConstantIndex(0);
    }

    ValueWithOffsets VisitPseudoObjectExpr(clang::PseudoObjectExpr* expr) {
        //if (auto syn = dyn_cast<MSPropertyRefExpr>(expr->getSyntacticForm ())) {
        //    return Visit(syn);
        //}
        return Visit(expr->getResultExpr());
    }

    ValueWithOffsets VisitUnaryOperator(UnaryOperator *U) {
        auto sub = Visit(U->getSubExpr());
        // TODO note assumptions made here about unsigned / unordered
        bool signedType = true;
        if (auto bit = dyn_cast<clang::BuiltinType>(&*U->getType())) {
            if (bit->isUnsignedInteger())
                signedType = false;
            if (bit->isSignedInteger())
                signedType = true;
        }
        auto ty = getMLIRType(&*U->getType());

        switch(U->getOpcode()) {
            case clang::UnaryOperator::Opcode::UO_PreInc:
            case clang::UnaryOperator::Opcode::UO_PostInc:{
                auto off = sub.offsets;
                if (off.size() == 0) {
                    off.push_back(getConstantIndex(0));
                }
                auto prev = mcg.builder_.create<mlir::LoadOp>(loc, sub.val, off);

                mlir::Value next;
                if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
                    next = mcg.builder_.create<mlir::AddFOp>(loc, prev, mcg.builder_.create<mlir::ConstantFloatOp>(loc, APFloat(ft.getFloatSemantics(), "1"), ft));
                } else {
                    next = mcg.builder_.create<mlir::AddIOp>(loc, prev, mcg.builder_.create<mlir::ConstantIntOp>(loc, 1, ty.cast<mlir::IntegerType>()));
                }
                mcg.builder_.create<mlir::StoreOp>(loc, next, sub.val, off);
                return ValueWithOffsets( (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PostInc) ? prev : next, {});
            }
            default:{defaultCase:
            U->dump();
            assert(0 && "unhandled opcode");
            }
        }
    }

    ValueWithOffsets VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr* expr) {
        return Visit(expr->getReplacement());
    }

    ValueWithOffsets VisitBinaryOperator(BinaryOperator *BO) {
        auto lhs = Visit(BO->getLHS());
        if (!lhs.val) {
            BO->getLHS()->dump();
        }
        assert(lhs.val);
        auto rhs = Visit(BO->getRHS());
        if (!rhs.val) {
            BO->getRHS()->dump();
        }
        assert(rhs.val);
        // TODO note assumptions made here about unsigned / unordered
        bool signedType = true;
        if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getType())) {
            if (bit->isUnsignedInteger())
                signedType = false;
            if (bit->isSignedInteger())
                signedType = true;
        }
        switch(BO->getOpcode()) {
            case clang::BinaryOperator::Opcode::BO_GT:{
                if (lhs.getType().isa<mlir::FloatType>()) {
                    return mcg.builder_.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::UGT, (mlir::Value)lhs, (mlir::Value)rhs);
                } else {
                    return mcg.builder_.create<mlir::CmpIOp>(loc, signedType ? mlir::CmpIPredicate::sgt : mlir::CmpIPredicate::ugt , (mlir::Value)lhs, (mlir::Value)rhs);
                }
            }
            case clang::BinaryOperator::Opcode::BO_GE:{
                if (lhs.getType().isa<mlir::FloatType>()) {
                    return mcg.builder_.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::UGE, (mlir::Value)lhs, (mlir::Value)rhs);
                } else {
                    return mcg.builder_.create<mlir::CmpIOp>(loc, signedType ? mlir::CmpIPredicate::sge : mlir::CmpIPredicate::uge, (mlir::Value)lhs, (mlir::Value)rhs);
                }
            }
            case clang::BinaryOperator::Opcode::BO_LT:{
                if (lhs.getType().isa<mlir::FloatType>()) {
                    return mcg.builder_.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::ULT, (mlir::Value)lhs, (mlir::Value)rhs);
                } else {
                    return mcg.builder_.create<mlir::CmpIOp>(loc, signedType ? mlir::CmpIPredicate::slt : mlir::CmpIPredicate::ult, (mlir::Value)lhs, (mlir::Value)rhs);
                }
            }
            case clang::BinaryOperator::Opcode::BO_LE:{
                if (lhs.getType().isa<mlir::FloatType>()) {
                    return mcg.builder_.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::ULE, (mlir::Value)lhs, (mlir::Value)rhs);
                } else {
                    return mcg.builder_.create<mlir::CmpIOp>(loc, signedType ? mlir::CmpIPredicate::sle : mlir::CmpIPredicate::ule, (mlir::Value)lhs, (mlir::Value)rhs);
                }
            }
            case clang::BinaryOperator::Opcode::BO_EQ:{
                if (lhs.getType().isa<mlir::FloatType>()) {
                    return mcg.builder_.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::UEQ, (mlir::Value)lhs, (mlir::Value)rhs);
                } else {
                    return mcg.builder_.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, (mlir::Value)lhs, (mlir::Value)rhs);
                }
            }
            case clang::BinaryOperator::Opcode::BO_NE:{
                if (lhs.getType().isa<mlir::FloatType>()) {
                    return mcg.builder_.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::UNE, (mlir::Value)lhs, (mlir::Value)rhs);
                } else {
                    return mcg.builder_.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ne, (mlir::Value)lhs, (mlir::Value)rhs);
                }
            }
            case clang::BinaryOperator::Opcode::BO_Mul:{
                if (lhs.getType().isa<mlir::FloatType>()) {
                    return mcg.builder_.create<mlir::MulFOp>(loc, (mlir::Value)lhs, (mlir::Value)rhs);
                } else {
                    return mcg.builder_.create<mlir::MulIOp>(loc, (mlir::Value)lhs, (mlir::Value)rhs);
                }
            }
            case clang::BinaryOperator::Opcode::BO_Div:{
                if (lhs.getType().isa<mlir::FloatType>()) {
                    return mcg.builder_.create<mlir::DivFOp>(loc, (mlir::Value)lhs, (mlir::Value)rhs);
                } else {
                    if (signedType)
                        return mcg.builder_.create<mlir::SignedDivIOp>(loc, (mlir::Value)lhs, (mlir::Value)rhs);
                    else
                        return mcg.builder_.create<mlir::UnsignedDivIOp>(loc, (mlir::Value)lhs, (mlir::Value)rhs);
                }
            }
            case clang::BinaryOperator::Opcode::BO_Rem:{
                if (lhs.getType().isa<mlir::FloatType>()) {
                    return mcg.builder_.create<mlir::RemFOp>(loc, (mlir::Value)lhs, (mlir::Value)rhs);
                } else {
                    if (signedType)
                        return mcg.builder_.create<mlir::SignedRemIOp>(loc, (mlir::Value)lhs, (mlir::Value)rhs);
                    else
                        return mcg.builder_.create<mlir::UnsignedRemIOp>(loc, (mlir::Value)lhs, (mlir::Value)rhs);
                }
            }
            case clang::BinaryOperator::Opcode::BO_Add:{
                if (lhs.getType().isa<mlir::FloatType>()) {
                    return mcg.builder_.create<mlir::AddFOp>(loc, (mlir::Value)lhs, (mlir::Value)rhs);
                } else {
                    return mcg.builder_.create<mlir::AddIOp>(loc, (mlir::Value)lhs, (mlir::Value)rhs);
                }
            }
            case clang::BinaryOperator::Opcode::BO_Sub:{
                if (lhs.getType().isa<mlir::FloatType>()) {
                    return mcg.builder_.create<mlir::SubFOp>(loc, (mlir::Value)lhs, (mlir::Value)rhs);
                } else {
                    return mcg.builder_.create<mlir::SubIOp>(loc, (mlir::Value)lhs, (mlir::Value)rhs);
                }
            }
            case clang::BinaryOperator::Opcode::BO_Assign:{
                auto off = lhs.offsets;
                if (off.size() == 0) {
                    off.push_back(getConstantIndex(0));
                }
                mcg.builder_.create<mlir::StoreOp>(loc, (mlir::Value)rhs, lhs.val, off);
                return rhs;
            }

            case clang::BinaryOperator::Opcode::BO_Comma:{
                return rhs;
            }


            case clang::BinaryOperator::Opcode::BO_AddAssign:{
                auto off = lhs.offsets;
                if (off.size() == 0) {
                    off.push_back(getConstantIndex(0));
                }
                auto prev = mcg.builder_.create<mlir::LoadOp>(loc, lhs.val, off);
                
                mlir::Value result;
                if (prev.getType().isa<mlir::FloatType>()) {
                    result = mcg.builder_.create<mlir::AddFOp>(loc, (mlir::Value)prev, (mlir::Value)rhs);
                } else {
                    result = mcg.builder_.create<mlir::AddIOp>(loc, (mlir::Value)prev, (mlir::Value)rhs);
                }
                mcg.builder_.create<mlir::StoreOp>(loc, result, lhs.val, off);
                return ValueWithOffsets(prev, {});
            }

            case clang::BinaryOperator::Opcode::BO_MulAssign:{
                auto off = lhs.offsets;
                if (off.size() == 0) {
                    off.push_back(getConstantIndex(0));
                }
                auto prev = mcg.builder_.create<mlir::LoadOp>(loc, lhs.val, off);
                
                mlir::Value result;
                if (prev.getType().isa<mlir::FloatType>()) {
                    result = mcg.builder_.create<mlir::MulFOp>(loc, (mlir::Value)prev, (mlir::Value)rhs);
                } else {
                    result = mcg.builder_.create<mlir::MulIOp>(loc, (mlir::Value)prev, (mlir::Value)rhs);
                }
                mcg.builder_.create<mlir::StoreOp>(loc, result, lhs.val, off);
                return ValueWithOffsets(prev, {});
            }

            default:{defaultCase:
            BO->dump();
            assert(0 && "unhandled opcode");
            }
        }
    }

    ValueWithOffsets VisitAttributedStmt(AttributedStmt *AS) {
        llvm::errs() << "warning ignoring attributes\n";
        return Visit(AS->getSubStmt());
    }

    ValueWithOffsets VisitDeclRefExpr (DeclRefExpr *E) {
        return getValue(E->getDecl()->getName().str());
    }

    ValueWithOffsets VisitOpaqueValueExpr(OpaqueValueExpr* E) {
        return Visit(E->getSourceExpr());
    }

   ValueWithOffsets VisitMemberExpr(MemberExpr *ME) {
       auto memberName = ME->getMemberDecl()->getName() ;
        llvm::errs() << "md name: " << memberName << "\n";
        if (auto sr2 = dyn_cast<OpaqueValueExpr>(ME->getBase())) {
            sr2->dump();
        if (auto sr = dyn_cast<DeclRefExpr>(sr2->getSourceExpr())) {
            sr->dump();
            if (sr->getDecl()->getName() == "blockIdx") {
                if (memberName == "__fetch_builtin_x") {
                }
                llvm::errs() << "known block index";
            }
            if (sr->getDecl()->getName() == "blockDim") {
                llvm::errs() << "known block dim";
            }
            if (sr->getDecl()->getName() == "threadIdx") {
                llvm::errs() << "known thread index";
            }
            if (sr->getDecl()->getName() == "gridDim") {
                llvm::errs() << "known grid index";
            }
        }}
        auto base = Visit(ME->getBase());
        return nullptr;
    }

    ValueWithOffsets VisitCastExpr(CastExpr *E) {
        switch(E->getCastKind()) {
            case clang::CastKind::CK_LValueToRValue:{
                if (auto dr = dyn_cast<DeclRefExpr>(E->getSubExpr())) {
                    if (dr->getDecl()->getName() == "warpSize") {
                        bool foundVal = false;
                        for(int i=scopes.size()-1; i>=0; i--) {
                            auto found = scopes[i].find("warpSize");
                            if (found != scopes[i].end()) {
                                foundVal = true;
                                break;
                            }
                        }
                        if (!foundVal) {
                            auto mlirType = getMLIRType(&*E->getType());
                            auto llvmType = getLLVMTypeFromMLIRType(mlirType);
                            return mcg.builder_.create<mlir::LLVM::DialectCastOp>(loc, mlirType,
                                mcg.builder_.create<mlir::NVVM::WarpSizeOp>(loc, llvmType));  
                        }
                    }
                }
                auto scalar = Visit(E->getSubExpr());
                auto off = scalar.offsets;
                if (off.size() == 0) {
                    off.push_back(getConstantIndex(0));
                }
                return mcg.builder_.create<mlir::LoadOp>(loc, scalar.val, off);
            }
            case clang::CastKind::CK_IntegralToFloating:{
                auto scalar = Visit(E->getSubExpr());
                auto ty = getMLIRType(&*E->getType()).cast<mlir::FloatType>();
                bool signedType = true;
                if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getType())) {
                    if (bit->isUnsignedInteger())
                        signedType = false;
                    if (bit->isSignedInteger())
                        signedType = true;
                }
                if (signedType)
                    return mcg.builder_.create<mlir::SIToFPOp>(loc, (mlir::Value)scalar, ty);
                else
                    return mcg.builder_.create<mlir::UIToFPOp>(loc, (mlir::Value)scalar, ty);
            }
            // TODO
            case clang::CastKind::CK_IntegralCast:{
                auto scalar = Visit(E->getSubExpr());
                return scalar;
            }
            case clang::CastKind::CK_ArrayToPointerDecay:{
                auto scalar = Visit(E->getSubExpr());
                auto mt = scalar.val.getType().cast<mlir::MemRefType>();
                auto shape2 = std::vector<int64_t>(mt.getShape());
                shape2[0] = -1;
                auto nex = mlir::MemRefType::get(shape2, mt.getElementType(), mt.getAffineMaps(), mt.getMemorySpace());
                return ValueWithOffsets(mcg.builder_.create<mlir::MemRefCastOp>(loc, scalar.val, nex), scalar.offsets);
            }
            case clang::CastKind::CK_FunctionToPointerDecay:{
                auto scalar = Visit(E->getSubExpr());
                return scalar;
            }
            default:
            E->dump();
            assert(0 && "unhandled cast");
        }
    }

    ValueWithOffsets VisitIfStmt(clang::IfStmt* stmt) {
        auto cond = (mlir::Value)Visit(stmt->getCond());
        assert(cond != nullptr);
    
        bool hasElseRegion = stmt->getElse();
        auto ifOp = mcg.builder_.create<mlir::scf::IfOp>(loc, cond, hasElseRegion);


        auto oldpoint = mcg.builder_.getInsertionPoint();
        auto oldblock = mcg.builder_.getInsertionBlock();
        mcg.builder_.setInsertionPointToStart(&ifOp.thenRegion().back());
        Visit(stmt->getThen());
        if (hasElseRegion) {
            mcg.builder_.setInsertionPointToStart(&ifOp.elseRegion().back());
            Visit(stmt->getElse());
        }

        mcg.builder_.setInsertionPoint(oldblock, oldpoint);
        return nullptr;
        //return ifOp;
    }


    ValueWithOffsets VisitCompoundStmt(clang::CompoundStmt* stmt) {
        for(auto a : stmt->children())
            Visit(a);
        return nullptr;
    }

    ValueWithOffsets VisitReturnStmt(clang::ReturnStmt* stmt) {
        if (stmt->getRetValue()) {
            auto rv = (mlir::Value)Visit(stmt->getRetValue());
            mcg.builder_.create<mlir::ReturnOp>(loc, rv);
        } else {
            mcg.builder_.create<mlir::ReturnOp>(loc);
        }
        return nullptr;
    }
};

struct MLIRASTConsumer : public ASTConsumer {
	Preprocessor &PP;
	ASTContext &ast_context;
	DiagnosticsEngine &diags;
	const char *function;
	set<ValueDecl *> live_out;
	codegen::MLIRCodegen &mcg;
    bool error;
    std::string fn;

	MLIRASTConsumer(std::string fn, Preprocessor &PP, ASTContext &ast_context,
		DiagnosticsEngine &diags, codegen::MLIRCodegen &mcg) :
		fn(fn), PP(PP), ast_context(ast_context), diags(diags),
		mcg(mcg), error(false)
	{
	}

	~MLIRASTConsumer() {
	}

	virtual bool HandleTopLevelDecl(DeclGroupRef dg) {
		DeclGroupRef::iterator it;

		if (error)
			return true;
		for (it = dg.begin(); it != dg.end(); ++it) {
			FunctionDecl *fd = dyn_cast<clang::FunctionDecl>(*it);
			if (!fd)
				continue;
            if (!fd->hasBody())
				continue;
            if (fd->getIdentifier() == nullptr) continue;
            if (!fd->isGlobal()) continue;
            //llvm::errs() << *fd << "  " << fd->isGlobal() << "\n";
            
            if (fd->getName() != fn) continue;
            
            MLIRScanner ms(fd, mcg);
            /*
			if (options->autodetect) {
				ScopLoc loc;
				pet_scop *scop;
				PetScan ps(PP, ast_context, fd, loc, options,
					    isl_union_map_copy(vb),
					    independent);
				scop = ps.scan(fd);
				if (!scop)
					continue;
				call_fn(scop);
				continue;
			}
			scan_scops(fd);
            */
		}

		return true;
	}
};

#include "llvm/Support/Host.h"
/* Helper function for ignore_error that only gets enabled if T
 * (which is either const FileEntry * or llvm::ErrorOr<const FileEntry *>)
 * has getError method, i.e., if it is llvm::ErrorOr<const FileEntry *>.
 */
template <class T>
static const FileEntry *ignore_error_helper(const T obj, int,
	int[1][sizeof(obj.getError())])
{
	return *obj;
}

/* Helper function for ignore_error that is always enabled,
 * but that only gets selected if the variant above is not enabled,
 * i.e., if T is const FileEntry *.
 */
template <class T>
static const FileEntry *ignore_error_helper(const T obj, long, void *)
{
	return obj;
}

/* Given either a const FileEntry * or a llvm::ErrorOr<const FileEntry *>,
 * extract out the const FileEntry *.
 */
template <class T>
static const FileEntry *ignore_error(const T obj)
{
	return ignore_error_helper(obj, 0, NULL);
}

/* Return the FileEntry corresponding to the given file name
 * in the given compiler instances, ignoring any error.
 */
static const FileEntry *getFile(CompilerInstance *Clang, std::string Filename)
{
	return ignore_error(Clang->getFileManager().getFile(Filename));
}

static void create_main_file_id(SourceManager &SM, const FileEntry *file)
{
	SM.setMainFileID(SM.createFileID(file, SourceLocation(),
					SrcMgr::C_User));
}


static void create_diagnostics(CompilerInstance *Clang)
{
	Clang->createDiagnostics();
}

#include "clang/Frontend/FrontendAction.h"
class MLIRAction : public clang::ASTFrontendAction {
    public:
    codegen::MLIRCodegen& mlir;
    std::string fn;
    MLIRAction(std::string fn, codegen::MLIRCodegen& mlir) : fn(fn), mlir(mlir) {

    }
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer (CompilerInstance &CI, StringRef InFile) override {
        return std::unique_ptr<clang::ASTConsumer> (new MLIRASTConsumer(fn, CI.getPreprocessor(), CI.getASTContext(), CI.getDiagnostics(), mlir));
    }
};

// -cc1 -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnu -S -disable-free -main-file-name saxpy.cu -mrelocation-model static -mframe-pointer=all -fno-rounding-math -fno-verbose-asm -no-integrated-as -aux-target-cpu x86-64 -fcuda-is-device -mlink-builtin-bitcode /usr/local/cuda/nvvm/libdevice/libdevice.10.bc -target-feature +ptx70 -target-sdk-version=11.0 -target-cpu sm_35 -fno-split-dwarf-inlining -debugger-tuning=gdb -v -resource-dir lib/clang/12.0.0 -internal-isystem lib/clang/12.0.0/include/cuda_wrappers -internal-isystem /usr/local/cuda/include -include __clang_cuda_runtime_wrapper.h -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/backward -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/backward -internal-isystem /usr/local/include -internal-isystem lib/clang/12.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu -internal-externc-isystem /include -internal-externc-isystem /usr/include -internal-isystem /usr/local/include -internal-isystem lib/clang/12.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu -internal-externc-isystem /include -internal-externc-isystem /usr/include -fdeprecated-macro -fno-dwarf-directory-asm -fno-autolink -fdebug-compilation-dir /mnt/Data/git/MLIR-GPU/build -ferror-limit 19 -fgnuc-version=4.2.1 -fcxx-exceptions -fexceptions -o /tmp/saxpy-a8baec.s -x cuda bin/saxpy.cu 

#include "clang/Frontend/TextDiagnosticBuffer.h"
static bool parseMLIR(const char *filename, std::string fn, std::vector<std::string> includeDirs, codegen::MLIRCodegen& mlir)
{
    std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());

    IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

    // Register the support for object-file-wrapped Clang modules.
    //auto PCHOps = Clang->getPCHContainerOperations();
    //PCHOps->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
    //PCHOps->registerReader(std::make_unique<ObjectFilePCHContainerReader>());

    // Buffer diagnostics from argument parsing so that we can output them using a
    // well formed diagnostic object.
    IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
    TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
    DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);
    //if (invocation)
    //    Clang->setInvocation(std::shared_ptr<CompilerInvocation>(invocation));
    bool Success;
    {
        const char *binary = "clang";
        const unique_ptr<Driver> driver(new Driver(binary, llvm::sys::getDefaultTargetTriple(), Diags));
        std::vector<const char *> Argv;
        Argv.push_back(binary);
        Argv.push_back(filename);
        Argv.push_back("--cuda-gpu-arch=sm_35");
        for(auto a : includeDirs) {
            Argv.push_back("-I");
            char* chars = (char*)malloc(a.length()+1);
            memcpy(chars, a.data(), a.length());
            chars[a.length()] = 0;
            Argv.push_back(chars);
        }
        
        const unique_ptr<Compilation> compilation(
            driver->BuildCompilation(llvm::ArrayRef<const char *>(Argv)));
        JobList &Jobs = compilation->getJobs();
        if (Jobs.size() < 1)
            return false;

        Command *cmd = cast<Command>(&*Jobs.begin());
        if (strcmp(cmd->getCreator().getName(), "clang"))
            return false;

        const ArgStringList *args = &cmd->getArguments();
        
        Success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(), *args, Diags);
    }
    Clang->getInvocation().getFrontendOpts().DisableFree = false;

    // Infer the builtin include path if unspecified.
    if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
        Clang->getHeaderSearchOpts().ResourceDir.empty())
      Clang->getHeaderSearchOpts().ResourceDir = LLVM_OBJ_ROOT "/lib/clang/" CLANG_VERSION_STRING;

    // Create the actual diagnostics engine.
    Clang->createDiagnostics();
    if (!Clang->hasDiagnostics())
      return false;

    DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
    if (!Success)
      return false;

    // Create and execute the frontend action.

    // Create the target instance.
   Clang->setTarget(TargetInfo::CreateTargetInfo(Clang->getDiagnostics(),
                                          Clang->getInvocation().TargetOpts));
   if (!Clang->hasTarget())
     return false;

   // Create TargetInfo for the other side of CUDA and OpenMP compilation.
   if ((Clang->getLangOpts().CUDA || Clang->getLangOpts().OpenMPIsDevice) &&
       !Clang->getFrontendOpts().AuxTriple.empty()) {
     auto TO = std::make_shared<clang::TargetOptions>();
     TO->Triple = llvm::Triple::normalize(Clang->getFrontendOpts().AuxTriple);
     TO->HostTriple = Clang->getTarget().getTriple().str();
     Clang->setAuxTarget(TargetInfo::CreateTargetInfo(Clang->getDiagnostics(), TO));
   }

   // Inform the target of the language options.
   //
   // FIXME: We shouldn't need to do this, the target should be immutable once
   // created. This complexity should be lifted elsewhere.
   Clang->getTarget().adjust(Clang->getLangOpts());

   // Adjust target options based on codegen options.
   Clang->getTarget().adjustTargetOptions(Clang->getCodeGenOpts(), Clang->getTargetOpts());
   


       MLIRAction Act(fn, mlir);

   for (const auto &FIF : Clang->getFrontendOpts().Inputs) {
     // Reset the ID tables if we are reusing the SourceManager and parsing
     // regular files.
     if (Clang->hasSourceManager() && !Act.isModelParsingAction())
       Clang->getSourceManager().clearIDTables();

     if (Act.BeginSourceFile(*Clang, FIF)) {
       
       llvm::Error err = Act.Execute();
       if (err) {
           llvm::errs() << "saw error: " << err << "\n";
         return false;
       }
       assert(Clang->hasSourceManager());



       Act.EndSourceFile();
       //llvm::errs() << "ended source file\n";
     }
   }
    return true;
    #if 0
	CompilerInstance *Clang = new CompilerInstance();
	create_diagnostics(Clang);
	DiagnosticsEngine &Diags = Clang->getDiagnostics();
	//Diags.setSuppressSystemWarnings(true);
	TargetInfo *target = create_target_info(Clang, Diags);
	Clang->setTarget(target);
	//set_lang_defaults(Clang);
	CompilerInvocation *invocation = construct_invocation(filename, Diags);
	if (invocation)
		set_invocation(Clang, invocation);
	//Diags.setClient(construct_printer(Clang, options->pencil));
	Clang->createFileManager();
	Clang->createSourceManager(Clang->getFileManager());
	HeaderSearchOptions &HSO = Clang->getHeaderSearchOpts();
	HSO.ResourceDir = LLVM_OBJ_ROOT "/lib/clang/" CLANG_VERSION_STRING;
	PreprocessorOptions &PO = Clang->getPreprocessorOpts();
	create_preprocessor(Clang);
	Preprocessor &PP = Clang->getPreprocessor();
	//add_predefines(PP, options->pencil);
	PP.getBuiltinInfo().initializeBuiltins(PP.getIdentifierTable(),
		PP.getLangOpts());

	const FileEntry *file = getFile(Clang, filename);
	//if (!file)
	//	isl_die(ctx, isl_error_unknown, "unable to open file",
	//		do { delete Clang; return isl_stat_error; } while (0));
	create_main_file_id(Clang->getSourceManager(), file);

	Clang->createASTContext();
	MLIRASTConsumer consumer(PP, Clang->getASTContext(), Diags,
				mlir);
	Sema *sema = new Sema(PP, Clang->getASTContext(), consumer);

    /*
	if (!options->autodetect) {
		PP.AddPragmaHandler(new PragmaScopHandler(scops));
		PP.AddPragmaHandler(new PragmaEndScopHandler(scops));
		PP.AddPragmaHandler(new PragmaLiveOutHandler(*sema,
							consumer.live_out));
	}
    */

	//consumer.add_pragma_handlers(sema);

	Diags.getClient()->BeginSourceFile(Clang->getLangOpts(), &PP);
	ParseAST(*sema);
	Diags.getClient()->EndSourceFile();

	delete sema;
	delete Clang;

	return consumer.error ? isl_stat_error : isl_stat_ok;
    #endif
}