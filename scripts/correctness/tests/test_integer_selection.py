"""Exact typed integer selection identities and rejection controls."""
from pathlib import Path
import sys, os, unittest
from unittest import mock
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import kernel_match as K


class IntegerSelection(unittest.TestCase):
    @mock.patch.dict(os.environ, {'POLYGEIST_NUMERIC_MODEL': 'strict'})
    def test_exact_typed_selection(self):
        K.configure_matcher('egglog')
        a,b=K.Term.In(0),K.Term.In(1);T=K.Term
        results=[]
        for cast in [None,'arith.extsi:i8 to i32','arith.extui:i8 to i32','arith.extsi:i16 to i32','arith.extui:i16 to i32']:
         x,y=(a,b) if cast is None else (T.Unary(cast,a),T.Unary(cast,b))
         for pred in ('slt','sle','ult','ule'):
          for low in (False,True):
           left=T.Select(T.Cmp(pred,x,y),a if low else b,b if low else a)
           right=T.Select(T.Cmp(pred,y,x),b if low else a,a if low else b)
           assert K.body_matches_template(left,right)=={},(cast,pred,low)
           wrong=T.Select(T.Cmp(pred,y,x),a if low else b,b if low else a)
           assert K.body_matches_template(left,wrong) is None,(cast,pred,low,'opposite selection accepted')
           results.append(dict(cast=cast,predicate=pred,low=low,positive=True,negative=True))
        for cast in ['arith.trunci:i32 to i8','arith.fptosi:f32 to i32']:
         x,y=T.Unary(cast,a),T.Unary(cast,b)
         assert K.body_matches_template(T.Select(T.Cmp('slt',x,y),a,b),T.Select(T.Cmp('slt',y,x),b,a)) is None
        x,y=T.Unary('arith.extsi:i8 to i32',a),T.Unary('arith.extui:i8 to i32',b)
        u,v=T.Unary('arith.extsi:i8 to i32',b),T.Unary('arith.extui:i8 to i32',a)
        assert K.body_matches_template(T.Select(T.Cmp('slt',x,y),a,b),T.Select(T.Cmp('slt',u,v),b,a)) is None
        assert K.body_matches_template(T.Select(T.Cmp('slt',a,b),a,b),T.Select(T.Cmp('ult',a,b),a,b)) is None
        for pred in ['olt','ole','ult_float']:
         assert K.body_matches_template(T.Select(T.Cmp(pred,a,b),a,b),T.Select(T.Cmp(pred,b,a),b,a)) is None
        # Exhaust all signed and unsigned byte values. The same proof depends only on
        # total integer order and injectivity, independent of wider integer widths.
        checks=0
        for signed in (False,True):
         values=range(-128,128) if signed else range(256)
         for x in values:
          for y in values:
           for nonstrict in (False,True):
            p=x<=y if nonstrict else x<y;q=y<=x if nonstrict else y<x
            assert (x if p else y)==(y if q else x)
            assert (y if p else x)==(x if q else y)
            checks+=2
        text='''linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%out : tensor<4xi8>) {
        ^bb0(%x: i8):
          linalg.yield %c-128_i8 : i8
        } -> tensor<4xi8>'''
        g=K.parse_generics('%c-128_i8 = arith.constant -128 : i8\n'+text,infer_outputs_from_yield=True)
        assert len(g)==1 and g[0].yield_values==['%c-128_i8'];assert str(K.encode_body(g[0]))==str(T.Lit(-128.0))


if __name__ == '__main__':
    unittest.main()
