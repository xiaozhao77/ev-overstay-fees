"""
Lightweight solver-independent modelling layer.

The model is first built in Python as variables plus linear constraints in matrix form and then passed to the chosen solver:
  'highs'  : open-source HiGHS (through scipy.optimize.milp), no installation or licence needed (default)
  'copt'   : COPT by Cardinal Operations (pip install coptpy, free academic licence)
  'gurobi' : Gurobi (pip install gurobipy, licence required)
"""
import math
import time
import numpy as np

INF = math.inf


class Expr:
    __slots__ = ('c', 'k')

    def __init__(self, c=None, k=0.0):
        self.c = {} if c is None else c     # {variable index: coefficient}
        self.k = float(k)                   # constant term

    def copy(self):
        return Expr(dict(self.c), self.k)

    @staticmethod
    def of(x):
        if isinstance(x, Expr):
            return x
        if isinstance(x, Var):
            return Expr({x.i: 1.0})
        return Expr(None, x)

    # ---- in-place addition and subtraction (for accumulation without repeated copying) ----
    def __iadd__(self, o):
        if isinstance(o, Var):
            self.c[o.i] = self.c.get(o.i, 0.0) + 1.0
        elif isinstance(o, Expr):
            for i, v in o.c.items():
                self.c[i] = self.c.get(i, 0.0) + v
            self.k += o.k
        else:
            self.k += o
        return self

    def __isub__(self, o):
        self += Expr.of(o) * -1.0
        return self

    # ---- ordinary operations (return a new object) ----
    def __add__(self, o):
        r = self.copy()
        r += o
        return r

    __radd__ = __add__

    def __sub__(self, o):
        r = self.copy()
        r += Expr.of(o) * -1.0
        return r

    def __rsub__(self, o):
        return Expr.of(o) - self

    def __neg__(self):
        return self * -1.0

    def __mul__(self, a):
        a = float(a)
        return Expr({i: v * a for i, v in self.c.items()}, self.k * a)

    __rmul__ = __mul__

    def __truediv__(self, a):
        return self * (1.0 / a)

    # ---- comparison -> constraint ----
    def __le__(self, o):
        return Constr(self - o, '<')

    def __ge__(self, o):
        return Constr(self - o, '>')

    def __eq__(self, o):
        return Constr(self - o, '=')

    __hash__ = object.__hash__


class Var:
    __slots__ = ('i', 'm')

    def __init__(self, i, m):
        self.i, self.m = i, m

    def _e(self):
        return Expr({self.i: 1.0})

    def __add__(self, o): return self._e() + o
    __radd__ = __add__
    def __sub__(self, o): return self._e() - o
    def __rsub__(self, o): return Expr.of(o) - self._e()
    def __neg__(self): return self._e() * -1.0
    def __mul__(self, a): return self._e() * a
    __rmul__ = __mul__
    def __truediv__(self, a): return self._e() / a
    def __le__(self, o): return self._e() <= o
    def __ge__(self, o): return self._e() >= o
    def __eq__(self, o): return self._e() == o
    __hash__ = object.__hash__

    @property
    def lb(self): return self.m._lb[self.i]

    @lb.setter
    def lb(self, v): self.m._lb[self.i] = float(v)

    @property
    def ub(self): return self.m._ub[self.i]

    @ub.setter
    def ub(self, v): self.m._ub[self.i] = float(v)

    @property
    def X(self): return self.m.value(self)


class Constr:
    __slots__ = ('e', 's')

    def __init__(self, e, s):
        self.e, self.s = e, s


def quicksum(it):
    r = Expr()
    for x in it:
        r += x
    return r


class Model:
    def __init__(self, name='model'):
        self.name = name
        self._lb, self._ub, self._int, self._names = [], [], [], []
        self.rows = []               # (dict, lb, ub)
        self.obj = Expr()
        self.maximize = True
        self.time_limit = 60.0
        self.mip_gap = 1e-4
        self.threads = 0
        self.verbose = False
        # results
        self.status = 'unsolved'
        self.sol = None
        self.obj_val = None
        self.obj_bound = None
        self.gap = None
        self.runtime = 0.0

    # ---- modelling ----
    def addVar(self, vtype='C', lb=0.0, ub=None, name=''):
        if ub is None:
            ub = 1.0 if vtype == 'B' else INF
        self._lb.append(float(lb))
        self._ub.append(float(ub))
        self._int.append(1 if vtype in ('B', 'I') else 0)
        self._names.append(name)
        return Var(len(self._lb) - 1, self)

    def addConstr(self, con):
        if isinstance(con, (bool, np.bool_)):
            if not con:
                raise ValueError('infeasible constant constraint')
            return
        e, s = con.e, con.s
        c = {i: v for i, v in e.c.items() if v != 0.0}
        rhs = -e.k
        if not c:                     # constant only
            ok = (s == '<' and 0 <= rhs + 1e-9) or (s == '>' and 0 >= rhs - 1e-9) or \
                 (s == '=' and abs(rhs) < 1e-9)
            if not ok:
                raise ValueError('infeasible constant constraint')
            return
        lo = rhs if s in ('>', '=') else -INF
        hi = rhs if s in ('<', '=') else INF
        self.rows.append((c, lo, hi))

    def setObjective(self, expr, maximize=True):
        self.obj = Expr.of(expr)
        self.maximize = maximize

    @property
    def NumVars(self):
        return len(self._lb)

    @property
    def has_solution(self):
        return self.sol is not None

    def value(self, x):
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        if isinstance(x, Var):
            return float(self.sol[x.i])
        if isinstance(x, Expr):
            return x.k + sum(v * self.sol[i] for i, v in x.c.items())
        raise TypeError(type(x))

    # ---- matrix form ----
    def _matrix(self):
        from scipy.sparse import csr_matrix
        n = self.NumVars
        data, rows, cols, lo, hi = [], [], [], [], []
        for r, (c, l, u) in enumerate(self.rows):
            for i, v in c.items():
                rows.append(r)
                cols.append(i)
                data.append(v)
            lo.append(l)
            hi.append(u)
        A = csr_matrix((data, (rows, cols)), shape=(len(self.rows), n))
        cvec = np.zeros(n)
        for i, v in self.obj.c.items():
            cvec[i] = v
        return A, np.array(lo), np.array(hi), cvec

    # ---- solve ----
    def optimize(self, solver='highs'):
        t0 = time.time()
        solver = (solver or 'highs').lower()
        A, lo, hi, c = self._matrix()
        if solver == 'highs':
            self._solve_highs(A, lo, hi, c)
        elif solver == 'copt':
            self._solve_copt(A, lo, hi, c)
        elif solver == 'gurobi':
            self._solve_gurobi(A, lo, hi, c)
        else:
            raise ValueError(f'unknown solver {solver}')
        self.runtime = time.time() - t0
        if self.sol is not None:
            self.sol = np.where(np.array(self._int) == 1, np.round(self.sol), self.sol)
            self.obj_val = self.value(self.obj)
        return self.sol is not None

    def _finish_gap(self):
        if self.obj_val is not None and self.obj_bound is not None and np.isfinite(self.obj_bound):
            self.gap = abs(self.obj_bound - self.obj_val) / max(1e-9, abs(self.obj_val))

    def _solve_highs(self, A, lo, hi, c):
        from scipy.optimize import milp, LinearConstraint, Bounds
        sign = -1.0 if self.maximize else 1.0
        cons = [LinearConstraint(A, lo, hi)] if A.shape[0] else []
        res = milp(sign * c, constraints=cons, integrality=np.array(self._int),
                   bounds=Bounds(np.array(self._lb), np.array(self._ub)),
                   options=dict(time_limit=self.time_limit, mip_rel_gap=self.mip_gap,
                                disp=self.verbose))
        self.status = {0: 'optimal', 1: 'time_limit', 2: 'infeasible', 3: 'unbounded'}.get(res.status, str(res.status))
        if res.x is not None:
            self.sol = np.asarray(res.x, dtype=float)
            self.obj_val = sign * res.fun + self.obj.k
            b = getattr(res, 'mip_dual_bound', None)
            if b is not None and np.isfinite(b):
                self.obj_bound = sign * b + self.obj.k
            elif res.status == 0:
                self.obj_bound = self.obj_val
            self._finish_gap()

    def _solve_copt(self, A, lo, hi, c):
        import coptpy as cp
        from coptpy import COPT
        if not hasattr(Model, '_copt_env'):
            Model._copt_env = cp.Envr()
        m = Model._copt_env.createModel(self.name)
        m.setParam(COPT.Param.Logging, 1 if self.verbose else 0)
        m.setParam(COPT.Param.LogToConsole, 1 if self.verbose else 0)
        m.setParam(COPT.Param.TimeLimit, self.time_limit)
        m.setParam(COPT.Param.RelGap, self.mip_gap)
        if self.threads:
            m.setParam(COPT.Param.Threads, self.threads)
        n = self.NumVars
        big = COPT.INFINITY
        x = [m.addVar(lb=self._lb[i], ub=min(self._ub[i], big), obj=float(c[i]),
                      vtype=COPT.BINARY if (self._int[i] and self._lb[i] >= 0 and self._ub[i] <= 1)
                      else (COPT.INTEGER if self._int[i] else COPT.CONTINUOUS)) for i in range(n)]
        A = A.tocsr()
        for r in range(A.shape[0]):
            s, e = A.indptr[r], A.indptr[r + 1]
            expr = cp.LinExpr()
            expr.addTerms([x[j] for j in A.indices[s:e]], A.data[s:e].tolist())
            l, u = lo[r], hi[r]
            m.addBoundConstr(expr, -big if not np.isfinite(l) else l, big if not np.isfinite(u) else u)
        m.setObjSense(COPT.MAXIMIZE if self.maximize else COPT.MINIMIZE)
        try:
            self._copt_solve_and_read(m, COPT)
        finally:
            m.dispose()          # release the COPT model promptly so that memory does not build up in long batch runs
            Model._copt_count = getattr(Model, '_copt_count', 0) + 1
            if Model._copt_count % 500 == 0:   # rebuild the COPT environment periodically
                Model._copt_env.close()
                del Model._copt_env

    def _copt_solve_and_read(self, m, COPT):
        m.solve()
        st = m.status
        self.status = {COPT.OPTIMAL: 'optimal', COPT.TIMEOUT: 'time_limit',
                       COPT.INFEASIBLE: 'infeasible'}.get(st, str(st))
        if m.hasmipsol if m.ismip else st == COPT.OPTIMAL:
            self.sol = np.array(m.getValues(), dtype=float)
            self.obj_val = m.objval + self.obj.k
            try:
                self.obj_bound = m.bestbnd + self.obj.k
            except Exception:
                self.obj_bound = None
            self._finish_gap()

    def _solve_gurobi(self, A, lo, hi, c):
        import gurobipy as gp
        from gurobipy import GRB
        m = gp.Model(self.name)
        m.Params.OutputFlag = 1 if self.verbose else 0
        m.Params.TimeLimit = self.time_limit
        m.Params.MIPGap = self.mip_gap
        if self.threads:
            m.Params.Threads = self.threads
        vt = np.array([GRB.BINARY if (self._int[i] and self._ub[i] <= 1) else
                       (GRB.INTEGER if self._int[i] else GRB.CONTINUOUS) for i in range(self.NumVars)])
        ub = np.array([u if np.isfinite(u) else GRB.INFINITY for u in self._ub])
        x = m.addMVar(self.NumVars, lb=np.array(self._lb), ub=ub, vtype=vt, obj=c)
        eq = np.isfinite(lo) & np.isfinite(hi) & (np.abs(lo - hi) < 1e-12)
        ge = np.isfinite(lo) & ~eq
        le = np.isfinite(hi) & ~eq
        for mask, sense, b in ((eq, '=', lo), (ge, '>', lo), (le, '<', hi)):
            if mask.any():
                m.addMConstr(A[mask], x, sense, b[mask])
        m.ModelSense = GRB.MAXIMIZE if self.maximize else GRB.MINIMIZE
        m.optimize()
        self._grb_read(m, x, GRB)
        m.dispose()

    def _grb_read(self, m, x, GRB):
        self.status = {GRB.OPTIMAL: 'optimal', GRB.TIME_LIMIT: 'time_limit',
                       GRB.INFEASIBLE: 'infeasible'}.get(m.Status, str(m.Status))
        if m.SolCount:
            self.sol = np.array(x.X, dtype=float)
            self.obj_val = m.ObjVal + self.obj.k
            self.obj_bound = m.ObjBound + self.obj.k
            self._finish_gap()
