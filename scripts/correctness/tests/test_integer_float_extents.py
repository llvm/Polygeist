"""Bound dimensions keep integer semantics through FP32/FP64 conversion."""
from pathlib import Path
import sys,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from region_match.affine import Aff
from region_match.equations import FConst,FIndex,FUn,fold_integer_casts
from region_match import matcher as M

class IntegerFloatExtents(unittest.TestCase):
    def cast(self,value,ty='f32',signed=True,bits=32):
        return FUn('arith.'+('sitofp' if signed else 'uitofp')+':i'+str(bits)+' to '+ty,FIndex(value))

    def test_constant_extent_is_a_candidate(self):
        self.assertEqual(M._solve_param(Aff.var('pn')-Aff.var('v0')-1,Aff.cst(63)-Aff.var('v0'),['pn']),('pn',Aff.cst(64)))
        self.assertIsNone(M._solve_param(Aff.var('pn')+Aff.var('v0'),Aff.var('v1')+3,['pn']))

    def test_fp32_rounding_is_preserved(self):
        f=self.cast(Aff.var('pn'))
        self.assertEqual(fold_integer_casts(f,{'pn':Aff.cst(2**24+1)}),FConst(2**24))
        self.assertEqual(fold_integer_casts(self.cast(Aff.cst(2**24+1),'f64')),FConst(2**24+1))

    def test_unbound_and_out_of_range_casts_remain_explicit(self):
        for f in [self.cast(Aff.var('pn')),self.cast(Aff.cst(-1),signed=False),self.cast(Aff.cst(32768),bits=16),self.cast(Aff.cst(2**53+1),bits=64)]:
            self.assertEqual(fold_integer_casts(f),f)

    def test_floating_precision_conversion_is_never_erased(self):
        f=FUn('arith.extf:f32 to f64',FConst(1.25))
        self.assertEqual(fold_integer_casts(f),f)

    def test_memory_cache_respects_semantic_key(self):
        # Switching model/body in one process must not reuse a prior model's
        # specification systems, even when disk caching is disabled.
        with patch.object(M,'_SPEC_CACHE',{}),patch.object(M,'_cache_key',side_effect=['strict','relaxed']),patch.object(M,'_load_specs_uncached',side_effect=[[],['new']]) as loader:
            self.assertEqual(M.load_specs('/unused',use_cache=False),[])
            self.assertEqual(M.load_specs('/unused',use_cache=False),['new'])
            self.assertEqual(loader.call_count,2)

if __name__=='__main__':unittest.main()
