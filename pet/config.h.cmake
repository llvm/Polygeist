/* config.h.in.  Generated from configure.ac by autoheader.  */

/* Define if HeaderSearchOptions::AddPath takes 4 arguments */
#define ADDPATH_TAKES_4_ARGUMENTS

/* Clang installation prefix */
#define CLANG_PREFIX @CLANG_PREFIX@

/* Define if CompilerInstance::createDiagnostics takes argc and argv */
/* #cmakedefine CREATEDIAGNOSTICS_TAKES_ARG */

/* Define if CompilerInstance::createPreprocessor takes TranslationUnitKind */
#define CREATEPREPROCESSOR_TAKES_TUKIND

/* Define if TargetInfo::CreateTargetInfo takes pointer */
/* #cmakedefine CREATETARGETINFO_TAKES_POINTER */

/* Define if TargetInfo::CreateTargetInfo takes shared_ptr */
#define CREATETARGETINFO_TAKES_SHARED_PTR

/* Define if CompilerInvocation::CreateFromArgs takes ArrayRef */
#define CREATE_FROM_ARGS_TAKES_ARRAYREF

/* Define if Driver constructor takes default image name */
/* #cmakedefine DRIVER_CTOR_TAKES_DEFAULTIMAGENAME */

/* Define to DiagnosticClient for older versions of clang */
/* #undef DiagnosticConsumer */

/* Define to Diagnostic for newer versions of clang */
#define DiagnosticInfo Diagnostic

/* Define to Diagnostic for older versions of clang */
/* #undef DiagnosticsEngine */

/* Define if getTypeInfo returns TypeInfo object */
#define GETTYPEINFORETURNSTYPEINFO

/* Define if llvm/ADT/OwningPtr.h exists */
/* #cmakedefine HAVE_ADT_OWNINGPTR_H */

/* Define if clang/Basic/DiagnosticOptions.h exists */
#define HAVE_BASIC_DIAGNOSTICOPTIONS_H

/* Define if getBeginLoc and getEndLoc should be used */
#define HAVE_BEGIN_END_LOC

/* Define if clang/Basic/LangStandard.h exists */
#define HAVE_CLANG_BASIC_LANGSTANDARD_H

/* Define if Driver constructor takes CXXIsProduction argument */
/* #cmakedefine HAVE_CXXISPRODUCTION */

/* Define if DecayedType is defined */
#define HAVE_DECAYEDTYPE

/* Define if SourceManager has findLocationAfterToken method */
#define HAVE_FINDLOCATIONAFTERTOKEN

/* Define if Driver constructor takes IsProduction argument */
/* #cmakedefine HAVE_ISPRODUCTION */

/* Define if clang/Lex/HeaderSearchOptions.h exists */
#define HAVE_LEX_HEADERSEARCHOPTIONS_H

/* Define if clang/Lex/PreprocessorOptions.h exists */
#define HAVE_LEX_PREPROCESSOROPTIONS_H

/* Define if llvm/Option/Arg.h exists */
#define HAVE_LLVM_OPTION_ARG_H

/* Define if SourceManager has a setMainFileID method */
#define HAVE_SETMAINFILEID

/* Define if StmtRange class is available */
/* #cmakedefine HAVE_STMTRANGE */

/* Define if SourceManager has translateLineCol method */
#define HAVE_TRANSLATELINECOL

/* Return type of HandleTopLevelDeclReturn */
#define HandleTopLevelDeclContinue true

/* Return type of HandleTopLevelDeclReturn */
#define HandleTopLevelDeclReturn bool

/* Define to Language::C or InputKind::C for newer versions of clang */
#define IK_C Language::C


/* Define to PragmaIntroducerKind for older versions of clang */
/* #undef PragmaIntroducer */

/* Defined if CompilerInstance::setInvocation takes a shared_ptr */
#define SETINVOCATION_TAKES_SHARED_PTR

/* Define if CompilerInvocation::setLangDefaults takes 5 arguments */
#define SETLANGDEFAULTS_TAKES_5_ARGUMENTS

/* Define to TypedefDecl for older versions of clang */
/* #undef TypedefNameDecl */

/* Define if Driver::BuildCompilation takes ArrayRef */
#define USE_ARRAYREF

/* Define to getHostTriple for older versions of clang */
/* #undef getDefaultTargetTriple */

/* Define to getInstantiationColumnNumber for older versions of clang */
/* #undef getExpansionColumnNumber */

/* Define to getInstantiationLineNumber for older versions of clang */
/* #undef getExpansionLineNumber */

/* Define to getInstantiationLoc for older versions of clang */
/* #undef getExpansionLoc */

/* Define to getLangOptions for older versions of clang */
/* #undef getLangOpts */

/* Define to getFileLocWithOffset for older versions of clang */
/* #undef getLocWithOffset */

/* Define to getResultType for older versions of clang */
/* #undef getReturnType */

/* Define to getTypedefForAnonDecl for older versions of clang */
/* #undef getTypedefNameForAnonDecl */

/* Define to InitializeBuiltins for older versions of clang */
/* #undef initializeBuiltins */
