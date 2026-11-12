"""Graph cuts retain exact expressions and reject unrepresentable stages."""
from pathlib import Path
import ctypes as C
import os
import random
import subprocess
import tempfile
import unittest

ROOT=Path(os.environ.get('POLYGEIST_ROOT',str(Path(__file__).resolve().parents[3])))
HEADER=Path(os.environ.get('POLYGEIST_PARTITION_HEADER',ROOT/'runtime/polygeist_pointwise_partition.h'))
ARITY={1:2,2:2,3:2,4:2,21:1,23:2,29:3,30:2,31:2,32:1}
class Stage(C.Structure):
    _fields_=[('words',C.c_uint64*12),('nodes',C.c_int),('external_count',C.c_int),('external',C.c_int*4),('output',C.c_int)]
class Plan(C.Structure):
    _fields_=[('count',C.c_int),('cuts',C.c_uint32),('used_inputs',C.c_uint),('stages',Stage*24)]
def encode(nodes):
    words=[0]*12
    for i,(op,args) in enumerate(nodes):
        value=op<<24
        for j,a in enumerate(args):value|=a<<(16-8*j)
        words[i//2]|=value<<(32*(i%2))
    return words

def expression(words,n,bindings):
    values=dict(bindings)
    for i in range(n):
        instruction=(words[i//2]>>(32*(i%2)))&0xffffffff
        op=instruction>>24
        values[12+i]=(op,tuple(values[(instruction>>s)&255] for s in [16,8,0][:ARITY[op]]))
    return values[11+n]

class PointwisePartition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        d=Path(cls.tmp.name)
        arity=''.join('case '+str(k)+':return '+str(v)+';' for k,v in ARITY.items())
        code='#include "'+str(HEADER)+'"\nstatic int arity(int op){switch(op){'+arity+'default:return -1;}}\nstatic int mul(int op){return op==2;}static int add(int op){return op==1||op==3;}\nstatic int separate(int op){return op==4;}static int boolean(int op){return (op>=23&&op<=28)||(op>=30&&op<=32); }\nint partition(const uint64_t*w,int n,struct pg_pointwise_partition*p){return pg_pointwise_partition_build_mixed(w,n,arity,mul,add,separate,boolean,p);}\n'
        (d/'test.c').write_text(code)
        subprocess.run(['gcc','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',str(d/'test.c'),'-o',str(d/'test.so')],check=True,capture_output=True)
        cls.lib=C.CDLL(str(d/'test.so'))
        cls.partition=cls.lib.partition
        cls.partition.argtypes=[C.POINTER(C.c_uint64),C.c_int,C.POINTER(Plan)]
        cls.partition.restype=C.c_int
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def run_plan(self,words,n):
        plan=Plan()
        return self.partition((C.c_uint64*12)(*words),n,C.byref(plan)),plan
    def verify(self,nodes):
        words=encode(nodes);status,plan=self.run_plan(words,len(nodes))
        if status<=0:return status
        base={i:('input' if i<4 else 'scalar',i) for i in range(12)}
        expected=expression(words,len(nodes),base);computed={}
        for j in range(plan.count):
            stage=plan.stages[j];bindings={i:base[i] for i in range(4,12)}
            for k in range(stage.external_count):
                ref=stage.external[k]
                bindings[k]=base[ref] if ref<4 else computed[ref-12]
            computed[stage.output]=expression(stage.words,stage.nodes,bindings)
            self.assertEqual(self.run_plan(stage.words,stage.nodes)[0],1 if stage.nodes==1 and ((stage.words[0]>>24)&255)==4 else 0)
        self.assertEqual(computed[plan.stages[plan.count-1].output],expected)
        return status
    def test_conditional_shared_expression(self):
        nodes=[(2,[0,4]),(2,[1,5]),(1,[12,13]),(23,[6,7]),(29,[15,14,2])]
        self.assertEqual(self.verify(nodes),1)
    def test_safe_fusion_and_dead_work_stay_unchanged(self):
        for nodes in [[(2,[0,4])],[(1,[0,4]),(2,[12,1])],[(2,[0,4]),(1,[12,1]),(1,[2,5])]]:
            self.assertEqual(self.verify(nodes),0)
    def test_invalid_or_wide_stage_rejected(self):
        for nodes in [[(2,[12,4])],[(255,[0])],[(2,[0,4]),(1,[99,1])]]:
            self.assertEqual(self.run_plan(encode(nodes),len(nodes))[0],-1)
        self.assertEqual(self.run_plan([0]*12,0)[0],-1)
        self.assertEqual(self.run_plan([0]*12,25)[0],-1)
        nodes=[(2,[i%4,4+i]) for i in range(5)]
        nodes += [(1,[12,13]),(1,[17,14]),(1,[18,15]),(1,[19,16])]
        self.assertEqual(self.run_plan(encode(nodes),len(nodes))[0],-1)
    def test_separate_backend_operands_are_materialized(self):
        nodes=[(1,[0,4]),(4,[1,12]),(3,[13,5]),(2,[14,2])]
        self.assertEqual(self.verify(nodes),1)
        status,plan=self.run_plan(encode(nodes),len(nodes))
        division=[s for s in plan.stages[:plan.count]if ((s.words[(s.nodes-1)//2]>>(32*((s.nodes-1)%2)+24))&255)==4]
        self.assertEqual(len(division),1)
        self.assertEqual(division[0].nodes,1)
    def test_dead_prefix_needs_no_intermediate_storage(self):
        nodes=[(2,[0,4]),(4,[1,2])]
        self.assertEqual(self.verify(nodes),1)
        status,plan=self.run_plan(encode(nodes),len(nodes))
        self.assertEqual(plan.count,1)
        self.assertEqual(plan.stages[0].nodes,1)
    def test_boolean_cannot_be_materialized_as_float(self):
        nodes=[(23,[0,4]),(4,[12,1])]
        self.assertEqual(self.run_plan(encode(nodes),len(nodes))[0],-1)
    def test_random_typed_dags_preserve_every_operand(self):
        randomizer=random.Random(20261112);partitioned=0
        for _ in range(500):
            floats=list(range(12));booleans=[];nodes=[]
            for i in range(randomizer.randrange(3,21)):
                op=randomizer.choice([1,2,2,3,4,21,23,29])
                if op==29 and not booleans:op=1
                args=([randomizer.choice(booleans),randomizer.choice(floats),randomizer.choice(floats)] if op==29 else [randomizer.choice(floats) for _ in range(ARITY[op])])
                nodes.append((op,args));(booleans if op==23 else floats).append(12+i)
            if nodes[-1][0]==23:nodes.append((29,[11+len(nodes),0,1]))
            nodes.append((2,[11+len(nodes),4]))
            nodes.append((1,[11+len(nodes),1]))
            partitioned+=self.verify(nodes)==1
        self.assertGreater(partitioned,100)
if __name__=='__main__':unittest.main()
