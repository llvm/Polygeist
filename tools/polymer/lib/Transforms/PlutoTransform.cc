//===- PlutoTransform.cc - Transform MLIR code by PLUTO -------------------===//
//
// This file implements the transformation passes on MLIR using PLUTO.
//
//===----------------------------------------------------------------------===//

#include "polymer/Transforms/PlutoTransform.h"
#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopStmtOpSet.h"
#include "polymer/Support/OslSymbolTable.h"
#include "polymer/Support/ScopStmt.h"
#include "polymer/Target/OpenScop.h"

#include "pluto/internal/pluto.h"
#include "pluto/osl_pluto.h"
#include "pluto/pluto.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;
using namespace llvm;
using namespace polymer;

/// The main function that implements the Pluto based optimization.
static mlir::FuncOp plutoTransform(mlir::FuncOp f, OpBuilder &rewriter) {
  PlutoContext *context = pluto_context_alloc();
  OslSymbolTable srcTable, dstTable;

  std::unique_ptr<OslScop> scop = createOpenScopFromFuncOp(f, srcTable);
  if (!scop)
    return nullptr;
  if (scop->getNumStatements() == 0)
    return nullptr;

  osl_scop_print(stderr, scop->get());

#if 0

  
  /* To tile or not? */
  int tile;
  /* Intra-tile optimization */
  int intratileopt;
  /* Diamond tiling for concurrent startup; enables concurrent startup along
   * one dimension. */
  int diamondtile;
  /* Use per connected component u and w instead of single u and w for the whole
   * program. */
  int per_cc_obj;
  /* Extract scop information from libpet*/
  int pet;
  /* Dynamic scheduling using Synthesized Runtime Interface. */
  int dynschedule;
  /* Dynamic scheduling - previous technique of building the entire task graph
   * in memory using Intel TBB Flow Graph scheduler */
  int dynschedule_graph;
  // Dynamic scheduling - previous technique of building the entire task graph
  // in memory using a custom DAG scheduler.
  // No longer maintained
  // TODO: remove this!
  int dynschedule_graph_old;
  /* consider transitive dependences between tasks */
  int dyn_trans_deps_tasks;
  /* Enables concurrent startup along dimensions  */
  int fulldiamondtile;
  /* Parallelization */
  int parallel;
  /* prefer pure inner parallelism to pipelined parallelism */
  int innerpar;
  /* Automatic unroll-jamming of loops */
  int unrolljam;
  /* unroll/jam factor */
  int ufactor;
  /* Enable or disable post-transformations to make code amenable to
   * vectorization (default - enabled) */
  int prevector;
  /* consider RAR dependences */
  int rar;
  /* Decides the fusion algorithm (MAXIMAL_FUSE, NO_FUSE, or SMART_FUSE) */
  FusionType fuse;
  /* For experimental purposes with dfp */
  int delayed_cut;
  /* Tyepd fuse at outer levels, max fuse at inner levels */
  int hybridcut;
  /* for debugging - print default cloog-style total */
  int scancount;
  /* parameters will be assumed to be at least this much */
  /* This is appended to the context passed to cloog */
  int codegen_context;
  /* Loop depth (1-indexed) to force as parallel */
  int forceparallel;
  /* multiple (currently two) degrees of pipelined parallelism */
  int multipar;
  /// Tile a second time for the next level of the memory hierarchy. By
  /// default tiling is done only for one level. A second level of tiling may
  /// in several cases reduce the number of tiles available for parallel
  /// execution.
  int second_level_tile;
  /* NOTE: --ft and --lt are to manually force tiling depths */
  /* First depth to tile (starting from 0) */
  int ft;
  /* Last depth to tile (indexed from 0)  */
  int lt;
  /* Output for debugging */
  int debug;
  /* More debugging output */
  int moredebug;
  /* Not implemented yet: Don't output anything unless something fails */
  int quiet;
  /* Identity transformation */
  int identity;
  /* Generate scheduling pragmas for Bee+Cl@k */
  int bee;
  /* Force this for cloog's -f */
  int cloogf;
  /* Force this for cloog's -l */
  int cloogl;
  /* Enable cloog's -sh (simple convex hull) */
  int cloogsh;
  /* Enable cloog's -backtrack */
  int cloogbacktrack;
  /* Use isl to compute dependences (default) */
  int isldep;
  /* Use candl to compute dependences */
  int candldep;
  /* Access-wise dependences with ISL */
  int isldepaccesswise;
  /* Coalesce ISL deps */
  int isldepcoalesce;
  /* Compute lastwriter for dependences */
  int lastwriter;
  /* DEV: Don't use cost function */
  int nodepbound;
  /* hard upper bound for transformation coefficients */
  int coeff_bound;
  /* Ask candl to privatize */
  int scalpriv;
  /* No output from Pluto if everything goes right */
  int silent;
  /* Read input from a .scop file */
  int readscop;
  /* Use PIP as the ILP solver. */
  int pipsolve;
  /* Use isl as the ILP solver. */
  int islsolve;
  /* Use glpk as the ILP solver. */
  int glpk;
  /* Use gurobi as the ILP solver. */
  int gurobi;
  /* Use lp instead of ILP. */
  int lp;
  /* Use pluto-(i)lp-dfp framework instead of pluto-ilp */
  int dfp;
  /* Use ILP with pluto-dfp instead of LP. */
  int ilp;
  /* Use LP solutions to colour SCCs */
  int lpcolour;
  /* Cluster the statements of the SCC. Currently supported with DFP based
   * approach only */
  int scc_cluster;
  /* Index set splitting */
  int iss;
  /* Output file name supplied from -o */
  char *out_file;
  /* Polyhedral compile time stats */
  int time;
  /* fast linear independence check */
  int flic;
#endif

  // Should use isldep, candl cannot work well for this case.

  // TODO: should discover why.
  context->options->isldep = 1;
  // context->options->isldepaccesswise = 0;
  // context->options->isldepcoalesce = 1;
  context->options->silent = 1;
  context->options->intratileopt = 0;
  context->options->diamondtile = 0;
  context->options->rar = 0;
  context->options->iss = 0;
  // context->options->fuse = kSmartFuse;
  // context->options->lastwriter = 0;
  context->options->identity = 1;
  context->options->tile = 1;
  context->options->parallel = 1;
  // context->options->multipar = 1;
  // context->options->second_level_tile = 1;
  // context->options->cloogsh = 1;
  // context->options->nodepbound = 1;
  // context->options->scalpriv = 1;

  pluto_options_alloc()

      PlutoProg *prog = osl_scop_to_pluto_prog(scop->get(), context);
  if (!context->options->silent) {
    fprintf(stderr, "[pluto] Number of statements: %d\n", prog->nstmts);
    // fprintf(stdout, "[pluto] Total number of loops: %d\n", dim_sum);
    fprintf(stderr, "[pluto] Number of deps: %d\n", prog->ndeps);
    fprintf(stderr, "[pluto] Maximum domain dimensionality: %d\n", prog->nvar);
    fprintf(stderr, "[pluto] Number of parameters: %d\n", prog->npar);
  }

  if (context->options->iss)
    pluto_iss_dep(prog);
  if (!context->options->identity)
    pluto_auto_transform(prog);
  pluto_compute_dep_directions(prog);
  pluto_compute_dep_satisfaction(prog);

  if (context->options->tile) {
    pluto_tile(prog);
  } else {
    if (context->options->intratileopt) {
      pluto_intra_tile_optimize(prog, 0);
    }
  }

  pluto_populate_scop(scop->get(), prog, context);
  osl_scop_print(stderr, scop->get());

  mlir::ModuleOp m = dyn_cast<mlir::ModuleOp>(f.getParentOp());
  mlir::FuncOp g = cast<mlir::FuncOp>(createFuncOpFromOpenScop(
      std::move(scop), m, dstTable, rewriter.getContext()));

  // Replace calls to f by g in every function within the whole module.

  pluto_context_free(context);
  return g;
}

namespace {
class PlutoTransformPass
    : public mlir::PassWrapper<PlutoTransformPass,
                               OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    mlir::OpBuilder b(m.getContext());

    SmallVector<mlir::FuncOp, 8> funcOps;
    llvm::DenseMap<mlir::FuncOp, mlir::FuncOp> funcMap;

    m.walk([&](mlir::FuncOp f) {
      if (!f.getAttr("scop.stmt"))
        funcOps.push_back(f);
    });

    for (mlir::FuncOp f : funcOps)
      if (mlir::FuncOp g = plutoTransform(f, b)) {
        funcMap[f] = g;
        g.setPrivate();
      }

    // Replacing the original scop top-level function with the pluto transformed
    // result, such that the whole end-to-end optimization is complete.
    m.walk([&](mlir::FuncOp f) {
      for (const auto &it : funcMap) {
        mlir::FuncOp from = it.first;
        mlir::FuncOp to = it.second;
        if (f != from)
          f.walk([&](mlir::CallOp op) {
            if (op.getCallee() == from.getName())
              op.setAttr("callee", b.getSymbolRefAttr(to.getName()));
          });
      }
    });
  }
};

} // namespace

void polymer::registerPlutoTransformPass() {
  PassRegistration<PlutoTransformPass>("pluto-opt",
                                       "Optimization implemented by PLUTO.");
}
