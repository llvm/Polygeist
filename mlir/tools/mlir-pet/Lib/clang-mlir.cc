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


struct MLIRScanner : public StmtVisitor<MLIRScanner, mlir::Value> {
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
        assert(0 && "couldnt find value");
    }

    mlir::Type getMLIRType(const clang::Type* t) {
        if (t->isSpecificBuiltinType(BuiltinType::Float)) {
            return mcg.builder_.getF32Type();
        }
        if (t->isSpecificBuiltinType(BuiltinType::Void)) {
            return mcg.builder_.getNoneType();
        }
        //if (auto pt = dyn_cast<clang::RecordType>(t)) {
        //    llvm::errs() << " thing: " << pt->getName() << "\n";
        //}
        t->dump();
        return nullptr;
    }

    mlir::Value createAllocOp(mlir::Type t, std::string name) {
        auto mr =mlir::MemRefType::get(1, t);
        auto alloc = mcg.builder_.create<mlir::AllocOp>(loc, mr);
        setValue(name, alloc);
        return alloc;
    }

    mlir::Value createAndSetAllocOp(std::string name, mlir::Value v) {
        auto op = createAllocOp(v.getType(), name);
        mlir::Value zeroIndex = mcg.builder_.create<mlir::ConstantIndexOp>(loc, 0);
        mcg.builder_.create<mlir::StoreOp>(loc, v, op, zeroIndex);
        return op;
    }

    MLIRScanner(FunctionDecl *fd, codegen::MLIRCodegen &mcg) : mcg(mcg), loc(mcg.builder_.getUnknownLoc()) {
        llvm::errs() << *fd << "\n";

        scopes.emplace_back();
        std::vector<mlir::Type> types;
        std::vector<std::string> names;
        for(auto parm : fd->parameters()) {
            types.push_back(getMLIRType(&*parm->getOriginalType()));
            names.push_back(parm->getName().str());
        }
        
        //auto argTypes = getFunctionArgumentsTypes(mcg.getContext(), inputTensors);
        auto funcType = mcg.builder_.getFunctionType(types, getMLIRType(&*fd->getReturnType()));
        mlir::FuncOp function(mlir::FuncOp::create(loc, fd->getName(), funcType));

        auto &entryBlock = *function.addEntryBlock();

        mcg.builder_.setInsertionPointToStart(&entryBlock);
        mcg.theModule_.push_back(function);

        for (unsigned i = 0, e = function.getNumArguments(); i != e; ++i) {
            //function.getArgument(i).setName(names[i]);
            createAndSetAllocOp(names[i], function.getArgument(i));
        }


        scopes.emplace_back();

        Stmt *stmt = fd->getBody();
        stmt->dump();
        Visit(stmt);
    }

    mlir::Value VisitBinaryOperator(BinaryOperator *BO) {
        auto lhs = Visit(BO->getLHS());
        auto rhs = Visit(BO->getRHS());
        switch(BO->getOpcode()) {
            case clang::BinaryOperator::Opcode::BO_GT:
                if (lhs.getType().isa<mlir::FloatType>()) {
                    return mcg.builder_.create<mlir::LLVM::FCmpOp>(loc, lhs.getType(), mlir::LLVM::FCmpPredicate::ugt, lhs, rhs);
                }
            default:
            BO->dump();
            assert(0 && "unhandled opcode");
        }
    }

    mlir::Value VisitDeclRefExpr (DeclRefExpr *E) {
        return getValue(E->getDecl()->getName().str());
    }

    mlir::Value VisitCastExpr(CastExpr *E) {
        auto scalar = Visit(E->getSubExpr());
        switch(E->getCastKind()) {
            case clang::CastKind::CK_LValueToRValue:{
                mlir::Value zeroIndex = mcg.builder_.create<mlir::ConstantIndexOp>(loc, 0);
                return mcg.builder_.create<mlir::LoadOp>(loc, scalar, zeroIndex);
            }
            default:
            E->dump();
            assert(0 && "unhandled cast");
        }
    }

    mlir::Value VisitIfStmt(clang::IfStmt* stmt) {
        auto cond = Visit(stmt->getCond());
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


    mlir::Value VisitCompoundStmt(clang::CompoundStmt* stmt) {
        for(auto a : stmt->children())
            Visit(a);
        return nullptr;
    }

    mlir::Value VisitReturnStmt(clang::ReturnStmt* stmt) {
        auto rv = stmt->getRetValue() ? Visit(stmt->getRetValue()) : nullptr;
        mcg.builder_.create<mlir::ReturnOp>(loc, rv);
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

	MLIRASTConsumer(Preprocessor &PP, ASTContext &ast_context,
		DiagnosticsEngine &diags, codegen::MLIRCodegen &mcg) :
		PP(PP), ast_context(ast_context), diags(diags),
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
            if (fd->getName() != "max") continue;
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
static TargetInfo *create_target_info(CompilerInstance *Clang,
	DiagnosticsEngine &Diags)
{
	shared_ptr<clang::TargetOptions> TO = Clang->getInvocation().TargetOpts;
	TO->Triple = llvm::sys::getDefaultTargetTriple();
	return TargetInfo::CreateTargetInfo(Diags, TO);
}

static void set_invocation(CompilerInstance *Clang,
	CompilerInvocation *invocation)
{
	Clang->setInvocation(std::shared_ptr<CompilerInvocation>(invocation));
}

static void set_lang_defaults(CompilerInstance *Clang)
{
	PreprocessorOptions &PO = Clang->getPreprocessorOpts();
	clang::TargetOptions &TO = Clang->getTargetOpts();
	llvm::Triple T(TO.Triple);
	CompilerInvocation::setLangDefaults(Clang->getLangOpts(), Language::CUDA, T, PO,
					    LangStandard::lang_unspecified);
}

static void create_preprocessor(CompilerInstance *Clang)
{
	Clang->createPreprocessor(TU_Complete);
}

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
    MLIRAction(codegen::MLIRCodegen& mlir) : mlir(mlir) {

    }
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer (CompilerInstance &CI, StringRef InFile) override {
        llvm::errs() << "creating consumer\n";
        return std::unique_ptr<clang::ASTConsumer> (new MLIRASTConsumer(CI.getPreprocessor(), CI.getASTContext(), CI.getDiagnostics(), mlir));
    }
};

// -cc1 -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnu -S -disable-free -main-file-name saxpy.cu -mrelocation-model static -mframe-pointer=all -fno-rounding-math -fno-verbose-asm -no-integrated-as -aux-target-cpu x86-64 -fcuda-is-device -mlink-builtin-bitcode /usr/local/cuda/nvvm/libdevice/libdevice.10.bc -target-feature +ptx70 -target-sdk-version=11.0 -target-cpu sm_35 -fno-split-dwarf-inlining -debugger-tuning=gdb -v -resource-dir lib/clang/12.0.0 -internal-isystem lib/clang/12.0.0/include/cuda_wrappers -internal-isystem /usr/local/cuda/include -include __clang_cuda_runtime_wrapper.h -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/backward -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/backward -internal-isystem /usr/local/include -internal-isystem lib/clang/12.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu -internal-externc-isystem /include -internal-externc-isystem /usr/include -internal-isystem /usr/local/include -internal-isystem lib/clang/12.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu -internal-externc-isystem /include -internal-externc-isystem /usr/include -fdeprecated-macro -fno-dwarf-directory-asm -fno-autolink -fdebug-compilation-dir /mnt/Data/git/MLIR-GPU/build -ferror-limit 19 -fgnuc-version=4.2.1 -fcxx-exceptions -fexceptions -o /tmp/saxpy-a8baec.s -x cuda bin/saxpy.cu 

#include "clang/Frontend/TextDiagnosticBuffer.h"
static bool parseMLIR(const char *filename, codegen::MLIRCodegen& mlir)
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
   


       MLIRAction Act(mlir);

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