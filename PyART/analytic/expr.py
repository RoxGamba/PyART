import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------
def _is_sympy(x):
    """Recursively check whether an object contains symbolic values."""
    if isinstance(x, (sp.Basic, sp.Expr, AnalyticExpression)):
        return True
    if isinstance(x, (list, tuple, np.ndarray)):
        return any(_is_sympy(i) for i in x)
    return False


def _get_x_exponent(term, x_symbol=None):
    """Return the exponent of *x_symbol* in a single expanded term."""
    if x_symbol is None:
        x_symbol = sp.symbols("x")
    elif isinstance(x_symbol, str):
        x_symbol = sp.symbols(x_symbol)
    elif not isinstance(x_symbol, sp.Symbol):
        raise TypeError("x_symbol must be str or sympy.Symbol")

    if term == 0:
        return sp.Integer(0)

    def walk(expr):
        if expr == x_symbol:
            return sp.Integer(1)
        if expr.is_Pow and expr.base == x_symbol:
            return sp.simplify(expr.exp)
        if expr.is_Mul:
            total = sp.Rational(0)
            for arg in expr.args:
                total += walk(arg)
            return total
        if expr.is_Add:
            powers = [walk(arg) for arg in expr.args]
            return max(powers)
        return sp.Integer(0)

    return walk(term)


def _get_x_power_range(expr, x_symbol=None):
    """Return (min_exponent, max_exponent) of *x_symbol* across all terms."""
    if x_symbol is None:
        x_symbol = sp.symbols("x")
    elif isinstance(x_symbol, str):
        x_symbol = sp.symbols(x_symbol)

    expr = sp.expand(expr)
    if expr.is_Add:
        powers = [_get_x_exponent(arg, x_symbol) for arg in expr.args]
        return min(powers), max(powers)
    else:
        p = _get_x_exponent(expr, x_symbol)
        return p, p


class MathDispatcher:
    """
    Unifies sympy and numpy calls to prevent code duplication
    in mathematical transformations.
    """

    def __init__(self, use_sympy):
        self.use_sympy = use_sympy
        self.sqrt = sp.sqrt if use_sympy else np.sqrt
        self.log = sp.log if use_sympy else np.log
        self.cos = sp.cos if use_sympy else np.cos
        self.sin = sp.sin if use_sympy else np.sin
        self.arctan2 = sp.atan2 if use_sympy else np.arctan2

    def dot(self, a, b):
        if len(a) != len(b):
            raise ValueError("Vectors must have the same dimension")
        if self.use_sympy:
            return sum(a[i] * b[i] for i in range(len(a)))
        return np.dot(a, b)

    def norm(self, v):
        if self.use_sympy:
            return self.sqrt(sum(x**2 for x in v))
        return np.linalg.norm(v)


class AnalyticExpression(object):
    """
    Parent class for analytic expressions, wrapping sympy for symbolic manipulation
    while offering high-performance, cached numerical evaluation via lazy lambdification.
    """

    def __init__(self, expr=None, var=None):
        self.expr = None if expr is None else sp.sympify(expr)

        # Auto-infer variables if none are provided
        if var is None and self.expr is not None:
            self.var = tuple(sorted(self.expr.free_symbols, key=lambda s: s.name))
        elif var is not None:
            self.var = tuple(
                sp.symbols(v) if isinstance(v, str) else v
                for v in (var if isinstance(var, (tuple, list)) else (var,))
            )
        else:
            self.var = tuple()

        self._compiled_func = None

    def __call__(self, *args, **kwds):
        if self.expr is None or not self.var:
            raise ValueError("No expression and/or variables set for evaluation")

        if args and not kwds:
            if len(args) != len(self.var):
                raise ValueError(f"Expected {len(self.var)} arguments, got {len(args)}")
            eval_args = args
        else:
            try:
                eval_args = tuple(kwds[v.name] for v in self.var)
            except KeyError as e:
                raise KeyError(f"Missing required variable: {e}")

        # Lazy lambdification: compile once, hit NumPy backend every time after
        if self._compiled_func is None:
            self._compiled_func = sp.lambdify(self.var, self.expr, modules="numpy")

        return self._compiled_func(*eval_args)

    def derivative(self, var):
        if self.expr is None:
            raise ValueError("No expression set")
        v = sp.symbols(var) if isinstance(var, str) else var
        return AnalyticExpression(self.expr.diff(v), var=self.var)

    def pn_order(self, var):
        if self.expr is None:
            raise ValueError("No expression set")
        v = sp.symbols(var) if isinstance(var, str) else var
        tmp = self.expr.replace(sp.log(v), sp.symbols("logx0"))
        tmp = sp.expand(tmp)
        min_exp, max_exp = _get_x_power_range(tmp, x_symbol=v)
        return max_exp, max_exp - min_exp

    def truncate(self, var, max_order):
        """
        Truncates the expression up to the target PN order using SymPy's series expansion.
        This is significantly faster and more robust than manual tree-walking.
        """
        if self.expr is None:
            raise ValueError("No expression set")
        v = sp.symbols(var) if isinstance(var, str) else var

        # Temporarily protect logs from series expansion
        log_v = sp.symbols("log_v_tmp")
        tmp = self.expr.replace(sp.log(v), log_v)

        # Perform algebraic truncation
        truncated = tmp.series(v, 0, max_order + 1).removeO()

        # Restore logs
        final_expr = truncated.replace(log_v, sp.log(v))
        return AnalyticExpression(final_expr, var=self.var)

    def to_latex(self):
        if self.expr is None:
            raise ValueError("No expression set")
        return sp.latex(self.expr)

    def to_function(self, modules="numpy"):
        if self.expr is None:
            raise ValueError("No expression set")
        if self._compiled_func is None:
            self._compiled_func = sp.lambdify(self.var, self.expr, modules=modules)
        return self._compiled_func, self.var

    # -------------
    # SymPy Methods
    # -------------

    def _sympy_(self):
        """Allows sympy functions (sp.sin, sp.expand) to interact natively."""
        return self.expr

    def __getattr__(self, name):
        if self.expr is not None and hasattr(self.expr, name):
            return getattr(self.expr, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __add__(self, other):
        other_expr = other.expr if isinstance(other, AnalyticExpression) else other
        return AnalyticExpression(self.expr + other_expr)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other_expr = other.expr if isinstance(other, AnalyticExpression) else other
        return AnalyticExpression(self.expr - other_expr)

    def __rsub__(self, other):
        other_expr = other.expr if isinstance(other, AnalyticExpression) else other
        return AnalyticExpression(other_expr - self.expr)

    def __mul__(self, other):
        other_expr = other.expr if isinstance(other, AnalyticExpression) else other
        return AnalyticExpression(self.expr * other_expr)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other_expr = other.expr if isinstance(other, AnalyticExpression) else other
        return AnalyticExpression(self.expr / other_expr)

    def __rtruediv__(self, other):
        other_expr = other.expr if isinstance(other, AnalyticExpression) else other
        return AnalyticExpression(other_expr / self.expr)

    def __pow__(self, other):
        other_expr = other.expr if isinstance(other, AnalyticExpression) else other
        return AnalyticExpression(self.expr**other_expr)
