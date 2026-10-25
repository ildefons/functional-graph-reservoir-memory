#!/usr/bin/env python3
"""Small integer exact signed functional-graph verification with SymPy."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from graph_exact_scaling import one_graph,graph_stats
import numpy as np
import sympy as sp
rng=np.random.default_rng(950317)
z=sp.Symbol('z');numbers=[]
for n in range(4,15):
    for k in range(4):
        sgn=None
        suc,sgn=one_graph(n,rng)
        stats=graph_stats(suc,sgn)
        P=sp.zeros(n,n)
        for i in range(n):P[i,int(suc[i])]=int(sgn[i])
        cycles=[tuple(int(x) for x in s.split(':')) for s in stats['cycle_specs'].split(';')]
        cyc_poly=sp.Poly(1,z)
        for L,eps in cycles:cyc_poly=sp.lcm(cyc_poly,sp.Poly(z**L-eps,z))
        m_poly=sp.Poly(z**stats['H'],z)*cyc_poly
        assert m_poly.degree()==stats['generic_krylov_rank_exact']
        coeffs=list(reversed(m_poly.all_coeffs()))
        Z=sp.zeros(n,n);Pow=sp.eye(n)
        for cf in coeffs:
            Z+=cf*Pow;Pow=Pow*P
        assert Z==sp.zeros(n,n),'annihilation failed'
        # Check degree is minimal with exact linearly independent powers (nullspace polynomial).
        powers=[];Pow=sp.eye(n)
        for j in range(m_poly.degree()+1):
            powers.append(sp.Matrix(n*n,1,list(Pow)));Pow=Pow*P
        assert sp.Matrix.hstack(*powers[:-1]).rank()==m_poly.degree(),'polynomial degree too large'
        assert sp.Matrix.hstack(*powers).rank()==m_poly.degree(),'annihilation rank mismatch'
        bs=sp.Matrix([int(x) for x in rng.integers(-200,200,size=n)])
        kry=[];v=bs
        for j in range(n):kry.append(v);v=P*v
        rk=sp.Matrix.hstack(*kry).rank()
        numbers.append((n,k,rk,m_poly.degree()))
        assert rk<=m_poly.degree()
print('Exact graph polynomial verified on',len(numbers),'independent graphs')
print('Generic Krylov rank match for random integer input',sum(rank==deg for n,k,rank,deg in numbers),'/',len(numbers))
print('Any exceptions',[(n,k,rank,deg) for n,k,rank,deg in numbers if rank!=deg])
