from sage.all import *
from typing import List, Tuple
from pprint import pprint
from time import time

# for faster computation
from sortedcontainers import SortedList

class BorderBasisCalculator:
    def __init__(self, ring):
        """Initialize with a polynomial ring."""
        self.ring = ring
        self.variables = ring.gens()

        # sorting time
        self.sorting_time = 0

    def terms_up_to_degree(self, d: int) -> List:
        """Compute all terms up to degree d."""
        n = len(self.variables)
        terms = [self.ring(1)]
        for t in range(1, d+1):
            exponents = list(WeightedIntegerVectors(t, [1]*n))
            terms.extend([self.ring.monomial(*e) for e in exponents])
            
        # terms = sorted(terms, key=lambda t: t)
        return terms
    
    def is_all_divisors_in(self, t: Polynomial, O: List) -> bool:
        """
        Check if adding term t to order ideal O maintains the order ideal property.
        
        Since O is already an order ideal, we only need to check immediate divisors
        (reducing degree by 1 for each variable).
        """
        if t.degree() in (0, 1): 
            return True
        
        return all([xi in O and t/xi in O for xi in self.variables if xi.divides(t)])

    def border(self, O: List) -> List:
        """
        Compute the border of an order ideal O.
        
        The border is the set of terms that are not in O, but have a divisor in O.
        """
        border = list()
        for t in O:
            for var in self.variables:
                new_term = t * var
                if new_term not in O and new_term not in border:
                    border.append(new_term)
        # border = sorted(border, key=lambda t: t.lm())
        return border
    
    def extend_V(self, V: List) -> List:
        V_plus = []
        for v in V:
            for var in self.variables:
                new_term = v * var
                V_plus.append(new_term)
        return V_plus

    def compute_border_basis(self, F: List, weights: dict, use_fast_elimination=False, lstabilization_only=False) -> Tuple[List, List]:
        """
        Implementation of Algorithm 4.1 (BBasis)
        
        Args:
            F: List of generating polynomials
            weights: Dictionary mapping terms to weights
            
        Returns:
            Tuple (G, O) where G is the border basis and O is the optimal order ideal
        """
        timings = {}  # Record execution time for each step
        
        # Step 1: Initial degree
        s = time()
        d = max(f.degree() for f in F)
        timings['step1_initial_degree'] = time() - s
        
        while True:
            # Step 2: Compute L-stable span
            s = time()
            L = self.terms_up_to_degree(d)
            M = self.compute_lstable_span(F, L, use_fast_elimination=use_fast_elimination)
            timings['step2_lstable_span'] = time() - s
            
            # Step 3: Check if universe is large enough
            s = time()
            terms_d = [t for t in L if t.degree() == d]
            lt_M = set(f.lm() for f in M)
            sufficient_universe = all(t in lt_M for t in terms_d)
            timings['step3_check_universe'] = time() - s
            
            if not sufficient_universe:
                d += 1
                continue
                
            # Step 4: Adjust dimension
            s = time()
            d_old = d
            d = len(self.terms_up_to_degree(d)) - len(M)
            if d <= d_old:
                M = [f for f in M if f.degree() <= d]
            else:
                M = self.compute_lstable_span(F, self.terms_up_to_degree(d), use_fast_elimination=use_fast_elimination)
            timings['step4_adjust_dimension'] = time() - s
            break
            
        if not lstabilization_only:
            # Step 5: Find optimal order ideal
            s = time()
            O = self.find_optimal_order_ideal(M, d, weights)
            timings['step5_find_order_ideal'] = time() - s
            
            # Step 6: Compute border basis
            s = time()
            G = self.basis_transformation(M, O, use_fast_elimination=use_fast_elimination)
            timings['step6_compute_basis'] = time() - s
            
            timings['total_time'] = sum(timings.values())
        
        else:
            O = []
            G = M
        
        print("\nExecution times:")
        print("-" * 40)
        for step, t in timings.items():
            print(f"{step:25}: {t:8.3f} seconds")
        print("-" * 40)
        
        return G, O, timings  # 

    def basis_transformation(self, M: List, O: List, use_fast_elimination=False) -> List:
        """
        Implementation of Algorithm 4.3 (BasisTransformation)
        
        Args:
            M: List of polynomials
            O: Order ideal (set of monomials)
        
        Returns:
            O-border basis G
        """
        if not M:
            return []
        
        # Step 1: Find maximum degree
        max_deg = max(p.degree() for p in M)
        
        # Sort M according to the ordering where O is an initial segment
        def term_key(p):
            lt = p.lm()
            return (lt not in O, lt) 
        
        M_sorted = sorted(M, key=term_key)

        # Step 2: Perform Gaussian elimination with the custom order
        Gprime = self.gaussian_elimination([], M_sorted, use_fast_elimination=use_fast_elimination)
        
        # Select polynomials with leading terms in border
        border = self.border(O)
        G = [g for g in Gprime if g.lm() in border]
        
        return G

    def gaussian_elimination_fast(self, V: List, G: List) -> List:
        """
        Optimized Gaussian elimination with binary search for matching leading terms in sorted `reducers`.

        Args:
            V: Sorted list of polynomials with distinct leading terms and normalized leading coefficients.
            G: List of polynomials to reduce (need not to be sorted).

        Returns:
            Sorted list of polynomials W satisfying conditions in Lemma 2.10.
        """
        assert (all([v.lc() == 1 for v in
                     V]))  # Ensure leading coefficients are 1 in V

        reducers =  SortedList(V, key=lambda p: p.lm())  # Copy of the initial sorted reduction base
        W = []  # Result list of reduced polynomials

        non_zero_reductions_indices = []

        from collections import defaultdict
        reduction_indices = defaultdict(list)

        # map newly added reducers to their indices
        reducer_indices = {}

        # iterate over polynomials in G
        for idx, f in enumerate(G):
            #f = H.pop(0)
            if f == 0:
                continue

            # Reduce f by binary searching for reducers with the same leading monomial
            reduced = True

            while reduced and f != 0:
                reduced = False
                f_lead_term = f.lm()

                # Find the reducer with the same leading monomial (binary search)
                index = reducers.bisect_key_left(f_lead_term)
                if index < len(reducers) and reducers[index].lm() == f_lead_term:
                    # Reduce f
                    reducer = reducers[index]
                    f = f - f.lc() * reducer
                    reduced = True
                    if index >= len(V):
                        reduction_indices[idx].append(reducer_indices[index])
            # If f is non-zero after reduction, normalize it, and add to reducers and W
            if f != 0:
                f = f / f.lc()  # Normalize leading coefficient to 1
                reducers.add(f)  # Insert into reducers, maintaining sorted order
                W.append(f)  # Append to W
                non_zero_reductions_indices.append(idx)

                # map newly added reducers to their indices
                reducer_indices[len(reducers)-1] = idx

                if not idx in reduction_indices.keys():
                    reduction_indices[idx] = []

        #print(f"Reduction indices: {reduction_indices}")
        #print(f"Reducer indices: {reducer_indices}")
        return W, non_zero_reductions_indices, reduction_indices

    def gaussian_elimination(self, V: List, G: List, use_fast_elimination=False) -> List:
        """
        Implementation of Algorithm 2.11 (GaussEl)
        
        Args:
            V: List of polynomials with distinct leading terms and normalized leading coefficients
            G: List of polynomials to reduce
            
        Returns:
            List of polynomials W satisfying conditions in Lemma 2.10
        """
        
        assert(all([v.lc() == 1 for v in V]))
        
        if use_fast_elimination:
            s = time()
            V = sorted(V, key=lambda p: p.lm())
            #print(f'V sorted in {time() - s:.3f} seconds')
            self.sorting_time += time()-s
            return self.gaussian_elimination_fast(V, G)
        
        reducers = list(V)  # Current reduction base
        H = list(G)  # Working copy
        
        while H:
            f = H.pop(0)
            if f == 0: continue
            
            # Reduce f by all available reducers
            reduced = True
            while reduced and f != 0:
                reduced = False
                for reducer in reducers:
                    if reducer.lm() == f.lm():
                        f = f - f.lc() * reducer
                        reduced = True
                        break  # Start over with new f
        
            if f != 0:
                f = f / f.lc()       # Normalize and add to results
                reducers.append(f)   # Add to reduction base
        
        W = reducers[len(V):]
        return W, None
    
    def compute_lstable_span(self, F: List, L: List, use_fast_elimination=False) -> List:
        """
        Implementation of Algorithm 2.13 (LStabSpan)
        
        Args:
            F: List of polynomials
            L: List of terms (computational universe)
            
        Returns:
            Vector space basis V of F^L with pairwise different leading terms
        """
        d = max(l.degree() for l in L)
        V = self.gaussian_elimination([], F, use_fast_elimination=use_fast_elimination)
        
        while True:
            # Reduce V+\V modulo V
            W = self.gaussian_elimination(V, self.border(V), use_fast_elimination=use_fast_elimination)
            
            # Keep only polynomials supported on L - this could also directly be done in Gaussian elimination
            W = [w for w in W if w.degree() <= d]
            
            if W:
                if use_fast_elimination:
                    # optional, here we could implement a merge step of two sorted lists
                    V.extend(W)
                else:
                    V.extend(W)
            else:
                break
            
        return V


    def find_optimal_order_ideal(self, M: List, d: int, weights: dict) -> List:
        """
        Implementation based on Lemma 3.9: Alternative description of order ideal polytope
        """
        from scipy.optimize import milp, Bounds, LinearConstraint
        from itertools import combinations
        import numpy as np
        
        # Get all terms up to degree d-1
        terms = self.terms_up_to_degree(d-1)
        n_terms = len(terms)
        
        # Objective coefficients
        c = np.array([-weights.get(term, 0) for term in terms])
        
        # All variables are binary
        integrality = np.ones(n_terms, dtype=np.int32)
        bounds = Bounds(0, 1)
        
        constraints = []
        
        timings = {}
        start = time()
        
        # Constraint (3.6a): Order ideal property
        order_rows = []
        for i, t1 in enumerate(terms):
            for j, t2 in enumerate(terms):
                if t1.divides(t2) and t1 != t2:
                    row = np.zeros(n_terms)
                    row[i] = -1
                    row[j] = 1
                    order_rows.append(row)
        
        if order_rows:
            A_order = np.vstack(order_rows)
            constraints.append(LinearConstraint(A_order, -np.inf, 0))
        
        # Constraint (3.6b): Size of order ideal is d
        A_eq = np.ones((1, n_terms))
        constraints.append(LinearConstraint(A_eq, d, d))
        
        # Get M_{≤d-1} and construct its coefficient matrix
        M_d_minus_1 = [f for f in M if f.degree() <= d-1]
        M_size = len(M_d_minus_1)
        
        # Construct the full coefficient matrix for M_{≤d-1}
        coeff_matrix = np.zeros((len(M_d_minus_1), len(terms)))
        for i, f in enumerate(M_d_minus_1):
            for j, term in enumerate(terms):
                coeff = f.monomial_coefficient(term)
                if coeff != 0:
                    coeff_matrix[i, j] = float(coeff)
        
        def get_submatrix_rank(U_indices):
            """
            Compute rank of the submatrix corresponding to columns in U_indices
            """
            return np.linalg.matrix_rank(coeff_matrix[:, U_indices])
        
        # Constraint (3.6c): sum(z_m for m in U) ≥ |U| - rk(Ũ)
        rank_rows = []
        rank_lbs = []
        
        total_combinations = sum(1 for _ in combinations(range(n_terms), M_size))
        # print(f"Total number of combinations to process: {total_combinations}")
                
        # iterator = tqdm(combinations(range(n_terms), M_size))
        iterator = combinations(range(n_terms), M_size)
        for U_indices in iterator:
            rank_U = get_submatrix_rank(U_indices)
            
            row = np.zeros(n_terms)
            row[list(U_indices)] = 1
            
            rank_rows.append(row)
            rank_lbs.append(M_size - rank_U)

        
        if rank_rows:
            A_rank = np.vstack(rank_rows)
            constraints.append(LinearConstraint(A_rank, rank_lbs, np.inf))
        # print(f"Added {len(rank_rows)} matrix rank constraints")
        
        # Solve MILP
        # print("Solving MILP...")
        options = {
            'disp': False,
            'presolve': True,
            'mip_rel_gap': 1e-4
        }
        
        timings['step1_setup_ineqs'] = time() - start
        start = time()

        try:
            result = milp(c=c, 
                        integrality=integrality,
                        bounds=bounds,
                        constraints=constraints,
                        options=options)
            
            timings['step2_optimization'] = time() - start
            
            print("\nExecution times:")
            print("-" * 40)
            for step, t in timings.items():
                print(f"{step:25}: {t:8.3f} seconds")
            print("-" * 40)
            
            if result.success:
                O = list(set(term for i, term in enumerate(terms) if result.x[i] > 0.5))
                O = sorted(O, key=lambda t: t)
                # print(f"Optimization successful. Objective value: {-result.fun}")
                return O
            else:
                print(f"Optimization failed: {result.message}")
                raise ValueError("Optimization failed")
                
        except Exception as e:
            print(f"Error solving MILP: {str(e)}")
            raise ValueError("Failed to find admissible order ideal")

if __name__ == '__main__':
    
    # dev params
    use_fast_elimination = True  # use gaussian_elimination_fast
    lstabilization_only = True   # only compute L-stable span
    
    # Define a polynomial ring
    R = PolynomialRing(QQ, 'x, y', order='degrevlex')
    x, y = R.gens()
    
    # Define a set of polynomials
    # Example 1
    d = 30  # F is 0-dim for any d; large d makes the computation slower
    F = [x**d *y - x - y, y**2 + x - 1]

    # Example 2
    # F = [x**2 - x*y, y**2 - x*y]
    # F += [t for t in BorderBasisCalculator(R).terms_up_to_degree(3) if t.degree() == 3]

    print('Input polynomials (F):')
    pprint(F)
    print()
    
    print(f'ideal dimension (should be 0): {ideal(F).dimension()}')
    print(f'Use fast gaussian elimination: {use_fast_elimination}')
    # Create a calculator
    calculator = BorderBasisCalculator(R)

    # Define a set of weights
    d = max(f.degree() for f in F)
    terms = calculator.terms_up_to_degree(d)
    weights = {t: 1 for i, t in enumerate(terms)}

    # Compute the border basis
    G, O, _ = calculator.compute_border_basis(F, weights, 
                                              use_fast_elimination=use_fast_elimination, 
                                              lstabilization_only=lstabilization_only)
    print("Sorting time: ", calculator.sorting_time)

    if lstabilization_only:
        print(f'Size of M: {len(G)}')
    else:    
        print('Border basis (G):')
        pprint(G)
        print()
        
        print('Optimal order ideal (O):')
        pprint(O)