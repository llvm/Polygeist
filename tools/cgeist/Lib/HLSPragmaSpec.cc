//===- HLSPragmaSpec.cc - Specifications of Vitis HLS pragmas -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The table of supported pragmas, their options, value types, defaults and
// placement. docs/hls-pragmas.md documents the same table; keep them in sync.
//
//===----------------------------------------------------------------------===//

#include "HLSPragmaInternal.h"

#include "llvm/ADT/STLExtras.h"

using namespace cgeist::hls;
using llvm::StringLiteral;
using llvm::StringRef;

namespace {

/// Small builder so the table below stays readable.
struct Opt {
  OptionSpec s;
  Opt(StringLiteral name, OptType type) { s.name = name, s.type = type; }
  Opt &values(std::initializer_list<StringLiteral> v) {
    s.values.assign(v.begin(), v.end());
    return *this;
  }
  Opt &required() { return s.required = true, *this; }
  Opt &positional() { return s.positional = true, *this; }
  Opt &min(int64_t m) { return s.min = m, *this; }
  Opt &key(StringLiteral k) { return s.attrKey = k, *this; }
  Opt &upper() { return s.upperCase = true, *this; }
  Opt &def(bool b) {
    s.defaultKind = OptionSpec::DefaultKind::Bool;
    s.defaultBool = b;
    return *this;
  }
  Opt &def(int64_t i) {
    s.defaultKind = OptionSpec::DefaultKind::Int;
    s.defaultInt = i;
    return *this;
  }
  Opt &def(StringLiteral str) {
    s.defaultKind = OptionSpec::DefaultKind::String;
    s.defaultString = str;
    return *this;
  }
  operator OptionSpec() const { return s; }
};

Opt flag(StringLiteral n) { return Opt(n, OptType::Flag); }
Opt integer(StringLiteral n) { return Opt(n, OptType::Int); }
Opt number(StringLiteral n) { return Opt(n, OptType::Number); }
Opt enumOpt(StringLiteral n, std::initializer_list<StringLiteral> v) {
  return Opt(n, OptType::Enum).values(v);
}
Opt variable() { return Opt("variable", OptType::Variable).required(); }
Opt name(StringLiteral n) { return Opt(n, OptType::Name); }

PragmaSpec pragma(StringLiteral name, Placement placement,
                  std::vector<OptionSpec> options, bool hasOff = false,
                  Multiplicity m = Multiplicity::Single) {
  PragmaSpec p;
  p.name = name;
  p.placement = placement;
  p.options = std::move(options);
  p.hasOff = hasOff;
  p.multiplicity = m;
  if (hasOff)
    p.options.push_back(flag("off"));
  return p;
}

std::vector<PragmaSpec> buildSpecs() {
  using P = Placement;
  using M = Multiplicity;
  std::vector<PragmaSpec> specs;

  // Partitioning options shared by array_partition and array_reshape.
  auto partitionOptions = [](bool reshape) {
    std::vector<OptionSpec> opts = {
        variable(),
        enumOpt("type", {"cyclic", "block", "complete"})
            .positional()
            .def(StringLiteral("complete"))
            .key("kind"),
        integer("factor").min(1),
        integer("dim").min(0).def(int64_t(1)),
    };
    if (reshape)
      opts.push_back(flag("object"));
    return opts;
  };

  //===-- Loop pragmas ---------------------------------------------------===//
  specs.push_back(pragma(
      "pipeline", P::LoopOrFunction,
      {integer("ii").min(1).key("II").def(int64_t(0)), flag("rewind").def(false),
       enumOpt("style", {"stp", "flp", "frp"}).def(StringLiteral("stp")),
       flag("enable_flush")},
      /*hasOff=*/true));
  specs.push_back(pragma("unroll", P::Loop,
                         {integer("factor").min(1).def(int64_t(0)),
                          flag("skip_exit_check").def(false)},
                         /*hasOff=*/true));
  specs.push_back(pragma("loop_flatten", P::Loop, {}, /*hasOff=*/true));
  specs.push_back(
      pragma("loop_merge", P::LoopOrFunction, {flag("force").def(false)}));
  specs.push_back(pragma("loop_tripcount", P::Loop,
                         {integer("min").min(0).def(int64_t(0)),
                          integer("max").min(0),
                          integer("avg").min(0)}));
  specs.push_back(pragma(
      "dependence", P::LoopOrFunction,
      {Opt("variable", OptType::Variable),
       enumOpt("class", {"array", "pointer"}).positional(),
       enumOpt("type", {"intra", "inter"})
           .positional()
           .def(StringLiteral("inter")),
       enumOpt("direction", {"raw", "war", "waw"}).positional().upper(),
       integer("distance").min(1),
       Opt("dependent", OptType::Flag).required()},
      false, M::Many));
  specs.push_back(pragma("dataflow", P::LoopOrFunction,
                         {flag("disable_start_propagation").def(false)}));
  specs.push_back(pragma("latency", P::LoopOrFunction,
                         {integer("min").min(0), integer("max").min(0)}));
  specs.push_back(pragma(
      "performance", P::LoopOrFunction,
      {number("target_ti"), number("target_tl"),
       enumOpt("unit", {"cycle", "sec"}).def(StringLiteral("cycle"))}));

  //===-- Function pragmas -----------------------------------------------===//
  specs.push_back(pragma("inline", P::Function, {flag("recursive").def(false)},
                         /*hasOff=*/true));
  // `name` may be left out: the enclosing function is the top (Vitis 2025.1
  // accepts `#pragma HLS top`; dsp ssr_fft tests main.cpp:29).
  specs.push_back(pragma("top", P::Function, {name("name")}));
  specs.push_back(pragma("function_instantiate", P::Variable, {variable()},
                         false, M::PerVariable));
  specs.push_back(
      pragma("expression_balance", P::LoopOrFunction, {}, /*hasOff=*/true));
  specs.push_back(pragma(
      "allocation", P::LoopOrFunction,
      {Opt("instances", OptType::NameList).required(),
       integer("limit").min(1).required(),
       enumOpt("type", {"function", "operation"})
           .positional()
           .def(StringLiteral("function"))},
      false, M::Many));
  specs.push_back(pragma("occurrence", P::LoopOrFunction,
                         {integer("cycle").min(1).required()}));
  specs.push_back(pragma("protocol", P::LoopOrFunction,
                         {enumOpt("mode", {"fixed", "floating"})
                              .positional()
                              .def(StringLiteral("floating"))}));
  specs.push_back(
      pragma("stable", P::Variable, {variable()}, false, M::PerVariable));
  specs.push_back(pragma("reset", P::Variable, {variable()}, /*hasOff=*/true,
                         M::PerVariable));

  //===-- Variable / port pragmas ----------------------------------------===//
  specs.push_back(pragma("array_partition", P::Variable, partitionOptions(false),
                         /*hasOff=*/true, M::PerVariable));
  specs.push_back(pragma("array_reshape", P::Variable, partitionOptions(true),
                         /*hasOff=*/true, M::PerVariable));
  specs.push_back(pragma(
      "bind_storage", P::Variable,
      {variable(),
       // xpm_memory: Vitis 2025.1 accepts it (Vitis_Libraries solver
       // potrf.hpp; kernel w9r22i_c3_xpm fails only on impl=uram for the
       // part, [HLS 214-370]).
       enumOpt("type", {"fifo", "ram_1p", "ram_1wnr", "ram_2p", "ram_s2p",
                        "ram_t2p", "rom_1p", "rom_2p", "rom_np",
                        "xpm_memory"}),
       enumOpt("impl", {"auto", "bram", "bram_ecc", "lutram", "uram",
                        "uram_ecc", "srl", "memory"}),
       integer("latency").min(-1)},
      false, M::PerVariable));
  // The legacy `RESOURCE variable=v core=C` (Vitis 2025.1 still accepts it):
  // a memory core is lowered as the bind_storage it names (HLSPragmaSema.cc,
  // lowerResourceCore).
  specs.push_back(pragma("resource", P::Variable,
                         {variable(), name("core").required(),
                          integer("latency").min(-1), name("ports"),
                          name("metadata")},
                         false, M::PerVariable));
  specs.push_back(pragma(
      "bind_op", P::Variable,
      {variable(),
       enumOpt("op", {"add",   "sub",   "mul",    "fadd",   "fsub",  "fmul",
                      "fdiv",  "fexp",  "flog",   "fsqrt",  "frsqrt", "frecip",
                      "dadd",  "dsub",  "dmul",   "ddiv",   "dexp",  "dlog",
                      "dsqrt", "drsqrt", "drecip", "hadd",  "hsub",  "hmul",
                      "hdiv",  "hsqrt"})
           .required(),
       enumOpt("impl", {"auto", "dsp", "fabric", "meddsp", "fulldsp", "maxdsp",
                        "primitivedsp", "qadder", "vivado_dds"}),
       integer("latency").min(-1)},
      false, M::PerVariable));
  specs.push_back(pragma(
      "interface", P::Port,
      {enumOpt("mode", {"ap_none", "ap_vld", "ap_ack", "ap_hs", "ap_ovld",
                        "ap_fifo", "ap_memory", "bram", "axis", "s_axilite",
                        "m_axi", "ap_ctrl_none", "ap_ctrl_hs", "ap_ctrl_chain",
                        "ap_stable"})
           .positional()
           .required(),
       Opt("port", OptType::Variable).required(), name("bundle"),
       Opt("offset", OptType::IntOrEnum).values({"off", "direct", "slave"}),
       integer("depth").min(1), integer("latency").min(0),
       integer("max_widen_bitwidth").min(0),
       integer("max_read_burst_length").min(1),
       integer("max_write_burst_length").min(1),
       integer("num_read_outstanding").min(1),
       integer("num_write_outstanding").min(1), flag("register"),
       enumOpt("register_mode", {"both", "forward", "reverse", "off"}),
       enumOpt("storage_type", {"ram_1p", "ram_2p", "ram_s2p", "ram_t2p",
                                "ram_1wnr", "rom_1p", "rom_2p", "rom_np"}),
       enumOpt("storage_impl", {"auto", "bram", "lutram", "uram"}),
       name("clock"), name("name"), integer("channel").min(0),
       integer("interrupt").min(0)},
      false, M::PerVariable));
  specs.push_back(pragma(
      "stream", P::Variable,
      {variable(), integer("depth").min(1).def(int64_t(2)),
       enumOpt("type", {"fifo", "pipo", "shared", "unsync"})
           .def(StringLiteral("fifo"))},
      /*hasOff=*/true, M::PerVariable));
  // `aggregate variable=x bit` (the bare compact mode: the open front end's
  // presentId("bit" | "byte" | "none" | "auto"), e.g. TAPA's vec.h).
  specs.push_back(pragma("aggregate", P::Variable,
                         {variable(), enumOpt("compact", {"bit", "byte", "none",
                                                          "auto"})
                                          .def(StringLiteral("auto"))
                                          .positional()},
                         false, M::PerVariable));
  specs.push_back(
      pragma("disaggregate", P::Variable, {variable()}, false, M::PerVariable));
  specs.push_back(pragma("alias", P::Function,
                         {Opt("ports", OptType::NameList).required(),
                          integer("distance"), Opt("offset", OptType::IntList)},
                         false, M::Many));
  specs.push_back(pragma("cache", P::Port,
                         {Opt("port", OptType::Variable).required(),
                          integer("lines").min(1), integer("depth").min(1)},
                         false, M::PerVariable));
  return specs;
}

const std::vector<PragmaSpec> &specTable() {
  static const std::vector<PragmaSpec> table = buildSpecs();
  return table;
}

} // namespace

const OptionSpec *PragmaSpec::findOption(StringRef lowerName) const {
  for (const OptionSpec &o : options)
    if (o.name == lowerName)
      return &o;
  return nullptr;
}

const OptionSpec *PragmaSpec::findPositional(StringRef lowerValue) const {
  for (const OptionSpec &o : options)
    if (o.positional && llvm::is_contained(o.values, lowerValue))
      return &o;
  return nullptr;
}

const PragmaSpec *cgeist::hls::lookupPragmaSpec(StringRef lowerName) {
  for (const PragmaSpec &p : specTable())
    if (p.name == lowerName)
      return &p;
  return nullptr;
}

llvm::ArrayRef<PragmaSpec> cgeist::hls::allPragmaSpecs() {
  return specTable();
}
