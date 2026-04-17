import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------
def _is_sympy(x):
    """Return whether an object contains SymPy-compatible symbolic values.

    Parameters
    ----------
    x : object
        Object or nested container to inspect.

    Returns
    -------
    bool
        ``True`` when ``x`` or one of its elements is symbolic.
    """
    if isinstance(x, (sp.Basic, sp.Expr, AnalyticExpression)):
        return True
    if isinstance(x, (list, tuple, np.ndarray)):
        return any(_is_sympy(i) for i in x)
    return False


def _get_x_exponent(term, x_symbol=None):
    """Return the exponent of a variable within a single expression term.

    Parameters
    ----------
    term : sympy.Expr
        Expression term whose power of ``x_symbol`` should be measured.
    x_symbol : str or sympy.Symbol, optional
        Variable whose exponent should be extracted. Defaults to ``x``.

    Returns
    -------
    sympy.Expr
        Exponent of ``x_symbol`` in ``term``.

    Raises
    ------
    TypeError
        If ``x_symbol`` is neither a string nor a SymPy symbol.
    """
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
    """Return the minimum and maximum powers of a variable in an expression.

    Parameters
    ----------
    expr : sympy.Expr
        Expression to inspect.
    x_symbol : str or sympy.Symbol, optional
        Variable whose exponents should be measured. Defaults to ``x``.

    Returns
    -------
    tuple[sympy.Expr, sympy.Expr]
        Minimum and maximum exponents of ``x_symbol`` across the expanded
        additive terms of ``expr``.
    """
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
    """Dispatch basic math operations to SymPy or NumPy.

    Parameters
    ----------
    use_sympy : bool
        If ``True``, symbolic SymPy operations are used. Otherwise NumPy
        numeric operations are used.
    """

    def __init__(self, use_sympy):
        """Initialize the dispatcher for symbolic or numeric evaluation.

        Parameters
        ----------
        use_sympy : bool
            Whether to route operations through SymPy.

        Returns
        -------
        None
            The dispatcher stores the selected backend functions.
        """
        self.use_sympy = use_sympy
        self.sqrt = sp.sqrt if use_sympy else np.sqrt
        self.log = sp.log if use_sympy else np.log
        self.cos = sp.cos if use_sympy else np.cos
        self.sin = sp.sin if use_sympy else np.sin
        self.arctan2 = sp.atan2 if use_sympy else np.arctan2

    def dot(self, a, b):
        """Return the dot product of two vectors.

        Parameters
        ----------
        a : sequence
            First vector.
        b : sequence
            Second vector.

        Returns
        -------
        sympy.Expr or float
            Dot product computed with the configured backend.

        Raises
        ------
        ValueError
            If the input vectors do not have the same length.
        """
        if len(a) != len(b):
            raise ValueError("Vectors must have the same dimension")
        if self.use_sympy:
            return sum(a[i] * b[i] for i in range(len(a)))
        return np.dot(a, b)

    def norm(self, v):
        """Return the Euclidean norm of a vector.

        Parameters
        ----------
        v : sequence
            Vector whose norm should be computed.

        Returns
        -------
        sympy.Expr or float
            Euclidean norm computed with the configured backend.
        """
        if self.use_sympy:
            return self.sqrt(sum(x**2 for x in v))
        return np.linalg.norm(v)


class AnalyticExpression(object):
    """Wrap a SymPy expression with helpers for analysis and evaluation.

    Parameters
    ----------
    expr : object, optional
        Expression that can be converted to a SymPy object.
    var : str, sympy.Symbol, sequence, or None, optional
        Variables used for numerical evaluation. When omitted, free symbols are
        inferred from ``expr``.
    """

    def __init__(self, expr=None, var=None):
        """Create an analytic expression wrapper.

        Parameters
        ----------
        expr : object, optional
            Expression that can be converted to a SymPy object.
        var : str, sympy.Symbol, sequence, or None, optional
            Evaluation variables. Free symbols are inferred when omitted.

        Returns
        -------
        None
            The expression, variables, and internal caches are stored on the
            instance.
        """
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
        # Cache completed truncation results keyed by variable and order.
        self._truncate_result_cache = {}
        # Cache the reusable term decomposition for the fast truncation path.
        self._truncate_term_cache = {}

    def __call__(self, *args, **kwds):
        """Evaluate the expression numerically using cached lambdification.

        Parameters
        ----------
        *args : tuple
            Positional values corresponding to ``self.var`` in order.
        **kwds : dict
            Keyword values keyed by variable name.

        Returns
        -------
        object
            Numerical result returned by the compiled backend function.

        Raises
        ------
        ValueError
            If the expression has no variables or the wrong number of
            positional arguments is supplied.
        KeyError
            If a required keyword argument is missing.
        """
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

    def _result_variables(self, expr):
        """Return an ordered variable tuple for a transformed expression.

        Parameters
        ----------
        expr : object
            Result of a symbolic transformation.

        Returns
        -------
        tuple[sympy.Symbol, ...] or None
            Variables aligned with the transformed expression, or ``None`` to
            trigger automatic inference.
        """
        if not isinstance(expr, sp.Basic):
            return None

        free_symbols = getattr(expr, "free_symbols", None)
        if free_symbols is None:
            return None

        if not self.var:
            return None

        current_vars = tuple(symbol for symbol in self.var if symbol in free_symbols)
        remaining_vars = tuple(
            sorted(
                free_symbols - set(current_vars),
                key=lambda symbol: symbol.name,
            )
        )
        return current_vars + remaining_vars

    def _wrap_result(self, result):
        """Wrap a SymPy result back into ``AnalyticExpression`` when possible.

        Parameters
        ----------
        result : object
            Result returned by a symbolic operation.

        Returns
        -------
        object
            Wrapped analytic expression for SymPy results, otherwise the raw
            result.
        """
        if not isinstance(result, sp.Basic):
            return result
        return AnalyticExpression(result, var=self._result_variables(result))

    def _apply_sympy_method(self, name, *args, **kwargs):
        """Call a SymPy expression method and preserve the wrapper.

        Parameters
        ----------
        name : str
            Name of the SymPy expression method to invoke.
        *args : tuple
            Positional arguments forwarded to the SymPy method.
        **kwargs : dict
            Keyword arguments forwarded to the SymPy method.

        Returns
        -------
        object
            Wrapped analytic expression when the SymPy method returns a
            symbolic expression, otherwise the raw method result.

        Raises
        ------
        ValueError
            If no expression is stored.
        """
        if self.expr is None:
            raise ValueError("No expression set")
        result = getattr(self.expr, name)(*args, **kwargs)
        return self._wrap_result(result)

    def derivative(self, var):
        """Differentiate the expression with respect to one variable.

        Parameters
        ----------
        var : str or sympy.Symbol
            Variable of differentiation.

        Returns
        -------
        AnalyticExpression
            Wrapped derivative of the stored expression.

        Raises
        ------
        ValueError
            If no expression is stored.
        """
        if self.expr is None:
            raise ValueError("No expression set")
        v = sp.symbols(var) if isinstance(var, str) else var
        return self._wrap_result(self.expr.diff(v))

    def pn_order(self, var):
        """Return the highest power and PN span of an expansion variable.

        Parameters
        ----------
        var : str or sympy.Symbol
            Expansion variable to inspect.

        Returns
        -------
        tuple[sympy.Expr, sympy.Expr]
            Highest exponent of ``var`` and the range between maximum and
            minimum exponents.

        Raises
        ------
        ValueError
            If no expression is stored.
        """
        if self.expr is None:
            raise ValueError("No expression set")
        v = sp.symbols(var) if isinstance(var, str) else var
        tmp = self.expr.replace(sp.log(v), sp.symbols("logx0"))
        tmp = sp.expand(tmp)
        min_exp, max_exp = _get_x_power_range(tmp, x_symbol=v)
        return max_exp, max_exp - min_exp

    @staticmethod
    def _extract_truncation_terms(expr, variable):
        """Return termwise powers of ``variable`` for fast truncation.

        Parameters
        ----------
        expr : sympy.Expr
            Expression to decompose into additive terms.
        variable : sympy.Symbol
            Expansion variable whose termwise exponent should be tracked.

        The fast truncation path only works when ``expr`` can be treated as a
        sum of terms whose dependence on ``variable`` is a single
        overall power.
        For each additive term, this helper computes that exponent and verifies
        that dividing the term by ``variable**exponent`` removes all remaining
        dependence on ``variable``.

        Returns
        -------
        list[tuple[sympy.Expr, sympy.Expr]] | None
            A list of ``(term, exponent)`` pairs when the expression is safe to
            truncate by exponent filtering, or ``None`` when a term has a
            symbolic exponent or retains non-factorable dependence on
            ``variable``.
        """
        terms = []
        for term in sp.Add.make_args(expr):
            exponent = _get_x_exponent(term, variable)
            if getattr(exponent, "free_symbols", None):
                return None

            try:
                remainder = term / (variable**exponent) if exponent != 0 else term
            except TypeError:
                return None

            if remainder.has(variable):
                return None

            terms.append((term, exponent))

        return terms

    def _build_truncation_terms(self, variable):
        """Prepare and cache the term data used by ``truncate``.

        Parameters
        ----------
        variable : sympy.Symbol
            Expansion variable used for truncation.

        This helper first protects ``log(variable)`` factors with a temporary
        placeholder so logarithms are preserved while powers of
        ``variable`` are analyzed. It then tries to extract termwise
        exponents directly from the stored expression and, if needed,
        from an ``expand_mul`` view that only expands multiplicative
        structure.

        Returns
        -------
        tuple[list[tuple[sympy.Expr, sympy.Expr]], sympy.Symbol] | None
            Cached ``(terms, log_placeholder)`` data for the fast truncation
            path, or ``None`` if the expression must fall back to the slower
            series-based path.
        """
        if variable in self._truncate_term_cache:
            return self._truncate_term_cache[variable]

        # Protect exact log(variable) factors so they survive truncation but
        # do not interfere with power counting.
        log_v = sp.Dummy(f"log_{variable.name}_tmp")
        protected_expr = self.expr.replace(sp.log(variable), log_v)

        terms = self._extract_truncation_terms(protected_expr, variable)
        if terms is None:
            # expand_mul exposes products of sums without the blow-up of a
            # full symbolic expand, which is usually enough for term filtering.
            expanded_expr = sp.expand_mul(protected_expr)
            if expanded_expr != protected_expr:
                terms = self._extract_truncation_terms(expanded_expr, variable)

        cache_entry = None if terms is None else (terms, log_v)
        self._truncate_term_cache[variable] = cache_entry
        return cache_entry

    def truncate(self, var, max_order):
        """
        Truncate the expression by removing terms whose exponent in ``var``
        is strictly greater than ``max_order``.

        When the expression is already a sum of termwise powers in the target
        variable, use cached exponent filtering. Fall back to SymPy series
        expansion plus the same exponent filtering for non-termwise
        expressions.

        In the fast path, exact ``log(var)`` factors are temporarily replaced
        by a placeholder symbol before exponent extraction. This makes
        logarithms behave like coefficients for power counting, so terms such
        as ``x**2 * log(x)`` and ``x**3 * log(x)**2`` are treated as orders
        ``2`` and ``3`` respectively. More complicated logarithmic dependence,
        such as ``log(1 + x)`` or ``log(x**2)``, is not normalized this way and
        will typically force the slower series-based fallback.

        Parameters
        ----------
        var : str or sympy.Symbol
            Variable used to define the truncation order.
        max_order : object
            Maximum power of ``var`` to retain.

        Returns
        -------
        AnalyticExpression
            Truncated analytic expression.

        Raises
        ------
        ValueError
            If no expression is stored.
        """
        if self.expr is None:
            raise ValueError("No expression set")
        v = sp.symbols(var) if isinstance(var, str) else var
        requested_order = sp.sympify(max_order)
        cache_key = (v, requested_order)

        # Repeated truncations at the same order are common in plotting and
        # benchmarking workflows, so memoize the wrapped result directly.
        if cache_key in self._truncate_result_cache:
            return self._truncate_result_cache[cache_key]

        term_cache = self._build_truncation_terms(v)
        if term_cache is not None:
            terms, log_v = term_cache
            try:
                # Fast path: keep only the terms whose overall power of v stays
                # within the requested cutoff.
                truncated_terms = [
                    term for term, exponent in terms if exponent <= requested_order
                ]
                truncated = (
                    sp.Add(*truncated_terms) if truncated_terms else sp.Integer(0)
                )
            except TypeError:
                term_cache = None

        if term_cache is None:
            # Protect logs before calling series so exact log(v) factors are
            # preserved in the reconstructed expression.
            log_v = sp.Dummy(f"log_{v.name}_tmp")
            tmp = self.expr.replace(sp.log(v), log_v)

            # SymPy's series truncation uses an integer order and rejects
            # negative values. Clamp at zero so Laurent-like negative-power
            # terms can still be generated and filtered below.
            series_order = max(0, int(sp.floor(requested_order)) + 1)
            truncated_series = tmp.series(v, 0, series_order).removeO()

            expanded_terms = self._extract_truncation_terms(
                sp.expand_mul(truncated_series),
                v,
            )
            if expanded_terms is None:
                truncated = truncated_series
            else:
                # Filter once more after the series call so fractional powers
                # above the requested cutoff are removed exactly.
                truncated_terms = [
                    term
                    for term, exponent in expanded_terms
                    if exponent <= requested_order
                ]
                truncated = (
                    sp.Add(*truncated_terms) if truncated_terms else sp.Integer(0)
                )

        # Restore the protected logarithms before wrapping the result.
        final_expr = truncated.replace(log_v, sp.log(v))
        result = self._wrap_result(final_expr)
        self._truncate_result_cache[cache_key] = result
        return result

    def subs(self, *args, **kwargs):
        """Apply SymPy substitution and preserve the wrapper when possible.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to :meth:`sympy.Expr.subs`.
        **kwargs : dict
            Keyword arguments forwarded to :meth:`sympy.Expr.subs`.

        Returns
        -------
        object
            Wrapped analytic expression for symbolic results, otherwise the
            raw substitution result.
        """
        return self._apply_sympy_method("subs", *args, **kwargs)

    def simplify(self, **kwargs):
        """Simplify the stored expression and preserve the wrapper.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments forwarded to :meth:`sympy.Expr.simplify`.

        Returns
        -------
        object
            Wrapped analytic expression for symbolic results, otherwise the
            raw simplification result.
        """
        return self._apply_sympy_method("simplify", **kwargs)

    def expand(self, *args, **kwargs):
        """Expand the stored expression and preserve the wrapper.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to :meth:`sympy.Expr.expand`.
        **kwargs : dict
            Keyword arguments forwarded to :meth:`sympy.Expr.expand`.

        Returns
        -------
        object
            Wrapped analytic expression for symbolic results, otherwise the
            raw expansion result.
        """
        return self._apply_sympy_method("expand", *args, **kwargs)

    def factor(self, *args, **kwargs):
        """Factor the stored expression and preserve the wrapper.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to :meth:`sympy.Expr.factor`.
        **kwargs : dict
            Keyword arguments forwarded to :meth:`sympy.Expr.factor`.

        Returns
        -------
        object
            Wrapped analytic expression for symbolic results, otherwise the
            raw factorization result.
        """
        return self._apply_sympy_method("factor", *args, **kwargs)

    def collect(self, *args, **kwargs):
        """Collect like terms in the stored expression.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to :meth:`sympy.Expr.collect`.
        **kwargs : dict
            Keyword arguments forwarded to :meth:`sympy.Expr.collect`.

        Returns
        -------
        object
            Wrapped analytic expression for symbolic results, otherwise the raw
            collect result.
        """
        return self._apply_sympy_method("collect", *args, **kwargs)

    def diff(self, *symbols, **kwargs):
        """Differentiate the stored expression and preserve the wrapper.

        Parameters
        ----------
        *symbols : tuple
            Differentiation variables and optional orders, as accepted by
            :meth:`sympy.Expr.diff`.
        **kwargs : dict
            Keyword arguments forwarded to :meth:`sympy.Expr.diff`.

        Returns
        -------
        object
            Wrapped analytic expression for symbolic results, otherwise the
            raw differentiation result.
        """
        normalized_symbols = tuple(
            sp.symbols(symbol) if isinstance(symbol, str) else symbol
            for symbol in symbols
        )
        return self._apply_sympy_method("diff", *normalized_symbols, **kwargs)

    def to_latex(self):
        """Render the stored expression as a LaTeX string.

        Returns
        -------
        str
            LaTeX representation of the stored expression.

        Raises
        ------
        ValueError
            If no expression is stored.
        """
        if self.expr is None:
            raise ValueError("No expression set")
        return sp.latex(self.expr)

    def to_function(self, modules="numpy"):
        """Compile the expression into a callable numerical function.

        Parameters
        ----------
        modules : str or sequence, optional
            Backend passed to :func:`sympy.lambdify`.

        Returns
        -------
        tuple[callable, tuple[sympy.Symbol, ...]]
            Compiled callable and the variable ordering it expects.

        Raises
        ------
        ValueError
            If no expression is stored.
        """
        if self.expr is None:
            raise ValueError("No expression set")
        if self._compiled_func is None:
            self._compiled_func = sp.lambdify(self.var, self.expr, modules=modules)
        return self._compiled_func, self.var

    # -------------
    # SymPy Methods
    # -------------

    def _sympy_(self):
        """Expose the wrapped SymPy expression to SymPy internals.

        Returns
        -------
        sympy.Expr
            Stored SymPy expression.
        """
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
