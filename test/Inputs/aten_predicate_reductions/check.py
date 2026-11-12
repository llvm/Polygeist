from pathlib import Path
import sys
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'scripts/correctness'))
import kernel_match_rewrite as km
base=Path(__file__).parent
cases=[('aten_count_nonzero_cpu','cubCountNonzero1D_f32_tensor'),('aten_equal_cpu','cubEqualAll1D_f32_tensor'),('aten_addr_elementwise','cudnnAddrElementwise_f32_memref')]
def rewrite(text):
    result=km.rewrite_mlir(text,enable_structured_rewrite=True)
    return result[0] if isinstance(result,tuple) else result
for name,symbol in cases:
    source=(base/(name+'.mlir')).read_text()
    output=rewrite(source)
    assert 'kernel.launch @'+symbol in output,(name,'missing valid match')
    if name=='aten_addr_elementwise':
        wrong=source.replace('arith.cmpf oeq','arith.cmpf une')
        assert 'kernel.launch @'+symbol not in rewrite(wrong),'wrong beta predicate accepted'
    else:
        import re
        wrong=re.sub(r'affine.store[^\n]*\n','',source)
        assert 'kernel.launch @'+symbol not in rewrite(wrong),'missing seed accepted'
        old,new=('constant 0 : i32','constant 2 : i32') if 'count' in name else ('constant 1 : i32','constant 0 : i32')
        assert 'kernel.launch @'+symbol not in rewrite(source.replace(old,new)),'wrong seed accepted'
        assert 'kernel.launch @'+symbol not in rewrite(source.replace('to offset: [0]','to offset: [1]')),'wrong output offset accepted'
        wrong=source.replace('arith.cmpf une','arith.cmpf oeq') if 'count' in name else source.replace('arith.cmpf oeq','arith.cmpf une')
        assert 'kernel.launch @'+symbol not in rewrite(wrong),'wrong input predicate accepted'
print('PASS: renamed inputs, identities, predicates and output views')
