from sage.all import (
    QQ,
    PolynomialRing,
    MixedIntegerLinearProgram,
    matrix,
    vector,
    var,
    solve,
)


class OAVI:
    """
    Symbolic computation of the order ideal and border basis of a set of points, implementing
    the Oracle Approximate Vanishing Ideal Algorithm from the paper:
    "Conditional Gradients for the Approximate Vanishing Ideal", by E.Wirth & S.Pokutta (2022).

    An exact oracle is employed. If a solution to the system A*x = b is not unique, the algorithm
    will return the first solution found.
    """
    def __init__(self, ring):
        """
        ring: A Sage PolynomialRing
        """
        self.ring = ring
        self.points = []
        self.O = [ring(1)]
        self.G = []

    def border(self, O, span_variables=None):
        if span_variables is None:
            span_variables = self.ring.gens()

        B = []
        for x in span_variables:
            B += [x * o for o in O if x * o not in O]
        return sorted(list(set(B)), key=lambda t: t)

    def init_points(self, points):
        """
        Inits points (given as a matrix) to the internal list.

        points: A Sage Matrix of shape (N, d), representing N points in d dimensions.
        """
        self.points.extend(points)

    def oracle(self, A, b):
        """
        Given a matrix A (m x n) and vector b (length m),
        solve A * x = b symbolically over variables x0, x1, ..., x_{n-1}.

        Returns a a solution vector x = [x0, x1, ...] if one exists.
        or raises an exception if no symbolic solution is found.
        """
        m, n = A.nrows(), A.ncols()

        # Create symbolic variables x0, x1, ..., x_{n-1}
        # We'll store them in a list: x_syms = [x0, x1, ...]
        x_syms = list(var('x%d' % i) for i in range(n))


        # Build the list of equations: A[i, :] * x_syms == b[i] for each row i
        eqns = []
        for i in range(m):
            lhs = sum(A[i, j]*x_syms[j] for j in range(n))
            eqns.append(lhs == b[i][0])

        # Solve the system
        solutions = solve(eqns, list(x_syms))


        # extract a solution vector from the dictionary
        if solutions:

            # The solution might contain free variables, so we substitute them with 0
            sol = []
            for eq in solutions[0]:
                rhs = eq.rhs()
                sol.append(rhs.subs({var: 0 for var in rhs.variables()}))

            return vector(QQ, sol)
        else:
            return None



    def evaluate_polynomial_terms(self, terms, points):
        """
        Evaluate a list of polynomials at each point in 'points'.

        terms:  A list of polynomials in the same ring (e.g. [x*y, x^2, y^2]).
        points: A Sage Matrix (N x d), each row is a point in d-dim space.

        Returns:
            A new matrix (N x len(terms)), whose (i, j)-th entry is
            terms[j] evaluated at points[i].
        """
        if points.nrows() == 0 or len(terms) == 0:
            return matrix(QQ, 0, 0)

        ring = terms[0].parent()
        gens = ring.gens()   # e.g. (x, y) for a 2D ring

        rows = []
        for i in range(points.nrows()):
            # Build substitution dict: { x: val_x, y: val_y, ... }
            subs_dict = {gens[j]: points[i, j] for j in range(points.ncols())}
            # Evaluate each polynomial at this point
            row_evals = [term.subs(subs_dict) for term in terms]
            rows.append(row_evals)

        return matrix(QQ, rows)

    def compute_border_basis(self):
        """
        Compute the border basis for the vanishing ideal of the points.
        """
        # compute border of O
        B = self.border(self.O)
        d = 1
        # terminate if B is empty
        while B:
            # evaluate the O_terms at the points
            O_eval = self.evaluate_polynomial_terms(self.O, matrix(QQ, self.points))
            for b in B:
                # evaluate b at the points
                b_eval = self.evaluate_polynomial_terms([b], matrix(QQ, self.points))
                # solve the system O_eval * x = b_eval
                solution = self.oracle(O_eval, b_eval)

                if solution is None:
                    # add b to the order ideal
                    self.O.append(b)
                else:
                    # extract the generator from the solution
                    g = b - sum(solution[i] * self.O[i] for i in range(len(solution)))
                    self.G.append(g)

            # extend the border
            B = self.border(self.O)
            d += 1

            # only keep the border terms of degree d
            B = [b for b in B if b.degree() == d]


        # return the order ideal and the border basis
        return self.O, self.G


# --------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Create a polynomial ring in two variables, say x, y over the rationals
    R = PolynomialRing(QQ, 2, names=('x, y'), order='degrevlex')
    x, y = R.gens()

    # 2) Instantiate the OAVI class
    oavi = OAVI(ring=R)

    # 3) Init rational points (example: 3 points in 2D)
    P = matrix(QQ, [[1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8]])
    oavi.init_points(P)



    O, G = oavi.compute_border_basis()
    print("Order ideal:")
    print(O)
    print("Border basis:")
    print(G)

    # check evaluation
    eval_result = oavi.evaluate_polynomial_terms(G, P)
    print("Evaluating border basis at points (Should be all zero matrix):")
    print(eval_result)

