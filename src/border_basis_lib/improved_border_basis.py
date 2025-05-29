# Standard library imports
from functools import cache
from typing import List, Tuple, Optional
from pprint import pprint
from bisect import bisect_left, insort
from time import time
from collections import defaultdict
from itertools import product, chain
import itertools
from collections import defaultdict

# Third-party imports
from mpmath.libmp.libelefun import exponential_series
from sage.all import *
from tqdm import tqdm
from sortedcontainers import SortedList

# Local imports
from src.border_basis_lib.border_basis import BorderBasisCalculator
from src.border_basis_lib.sanity_checks import OBorderBasisChecker
from src.border_basis_lib.utils import (
    plot_multiple_monomials,
    find_maximal_terms,
    collect_all_indices,
    compute_index_and_expansion_direction,
    expand_within_universe
)
from src.oracle import (
    Oracle,
    HardPartialExtensionOracle,
    HistoricalOracle,
    TransformerOracle
)

from functools import cache

from mpmath.libmp.libelefun import exponential_series
from sage.all import *
from typing import List, Tuple, Optional
from pprint import pprint
from bisect import bisect_left, insort
from tqdm import tqdm
from time import time
from sortedcontainers import SortedList
from itertools import product

from .border_basis import BorderBasisCalculator
from .sanity_checks import OBorderBasisChecker

from .utils import plot_multiple_monomials, find_maximal_terms, collect_all_indices, compute_index_and_expansion_direction, expand_within_universe

# from ..oracle import Oracle, HardPartialExtensionOracle, HistoricalOracle


class ImprovedBorderBasisCalculator(BorderBasisCalculator):
    """
    Implementation of the Improved Border Basis algorithm from the paper "Computing Border Bases" by Kehrein & Kreuzer (2005).

    Saves successful expansion directions to train an oracle for their prediction. The datasets are created as lists of tuples (L, V, successful_expansion_directions). L is in short form, i.e. a list of monomials that spans the same order ideal as the full order ideal.

    The first index of self.datasets corresponds to the call of L-stable span computation. The second index corresponds to the iteration of the L-stable span computation.

    Example:
    self.datasets[0][0] = [[z**2, y*z, y**2, x**4*z, x**4*y, x**5], [z**2 + 3*y - 7*z, y*z - 4*y, x*z - 4*y, y**2 - 4*y, x*y - 4*y, x**5 - 8*x**4 + 14*x**3 + 8*x**2 - 15*x + 15*y], [x**2*z - 4*x*y, x**2*y - 4*x*y], [('x', 2), ('x', 4)]]

    Here we consider the first call of L-Stable-Span and the first iteration (i.e. the first extension of V).

    L is the order ideal spanned by [z**2, y*z, y**2, x**4*z, x**4*y, x**5].

    V is the set of polynomials that we aim to extend, here:
    [z**2 + 3*y - 7*z, y*z - 4*y, x*z - 4*y, y**2 - 4*y, x*y - 4*y, 
     x**5 - 8*x**4 + 14*x**3 + 8*x**2 - 15*x + 15*y]

    The successful expansion directions are [x**2*z - 4*x*y, x**2*y - 4*x*y].

    The expansion directions are [('x', 2), ('x', 4)].
    
    That is we extended:
    - The third polynomial in V which is x*z - 4*y in x-direction
    - The fifth polynomial in V which is x*y - 4*y in x-direction.
    """
    def __init__(self,
        ring,
        corollary_23: bool = False,
        N: int = 1,
        save_universes: bool = False,
        save_expansion_directions: bool = True,
        L: Optional[List] = None,
        verbose: bool = False,
        sorted_V: bool = False,
        relative_gap: float = 0.5,
        absolute_gap: int = 100,
        order_ideal_size: int = 5,
        oracle_max_calls: int = 5,
        min_universe: int = 20,
        save_path: Optional[str] = None,
        leading_term_k: int = 5,
        oracle: bool = False,
        oracle_model: Optional[TransformerOracle] = None):
        """
        Initialize the ImprovedBorderBasisCalculator with the polynomial ring.
        """
        super().__init__(ring)
        # whether to use corollary 23 from the computing border basis paper
        self.corollarly_23 = corollary_23
        self.logging_enabled = False
        self.verbose = verbose
        self.count = 1
        # iterations to do full universe expansion
        self.N = N

        # whether to sort the set V
        self.sorted_V = sorted_V

        self.L = L

        # oracle hyperparameters
        self.absolute_gap = absolute_gap
        self.relative_gap = relative_gap
        self.order_ideal_size = order_ideal_size
        self.oracle_max_calls = oracle_max_calls
        self.min_universe = min_universe



        # Initialize the TransformerOracle if oracle is True and oracle_model is not provided
        if oracle and oracle_model is None:
            from border_basis_lib.transformer_oracle import TransformerOracle
            self.oracle = TransformerOracle(ring, save_path, leading_term_k=leading_term_k)
        else:
            self.oracle = oracle_model

        self.oracle_calls = 0
        self.use_oracle = oracle

        # loggin params 
        self.save_universes = save_universes
        self.universes = []
        self.leading_terms = []
        self.border_terms_add = []
        self.save_expansion_directions = save_expansion_directions

        # Save the successful expansion directions in the order they were added
        self.successful_expansion_directions = []

        self.surviving_indices = []

        # Save the datasets - one per call of L-stable span computation
        self.datasets = []

        # Saved the actual expansions vs the total expansions
        self.efficiency = []

        # timings and other metrics
        self.timings = {}

        self.timings['step1_initial_order_ideal'] = 0
        self.timings['step2_lstable_span_improved'] = []
        # time taken for gaussian eliminations
        self.timings['gaussian_elimination_times'] = []
        self.timings['step3_check_universe'] = 0
        # legacy
        self.timings['total_efficiency'] = 0
        self.timings['total_time'] = 0
        # total number of reductions = gaussian eliminations
        self.timings['total_reductions'] = 0
        # counts how many times we fall back to border basis algorithm
        self.timings['fallback_to_border_basis'] = 0
        
        
    def enable_logging(self, enabled: bool):
        """Enable or disable logging for debugging and performance tracking."""
        self.logging_enabled = enabled

    def log(self, message: str):
        """Log a message if logging is enabled."""
        if self.logging_enabled:
            print(f"[LOG]: {message}")

    def border(self, V: List, origin = False) -> List:
        """
        Implementation for extending the set V to all directions, notes also the expansion origin.

        For origin = True, it returns the expansion origin.
        Example: V = [x*+2 + y, y], border(V) = [(x*+2 + y, x**3+x*y), (x*+2 + y, x**2*y+y**2), (y, x*y), (y, y**2)].

        For order ideals it returns the border of the order ideal.

        Args:
            V: List of polynomials (or monomials) in the polynomial ring.
            origin: Flag to include expansion origin.
        Returns:
            List: The border of the set V and the expansion origin.
        """
        border = list()
        for t in O:
            for var in self.variables:
                new_term = t * var
                if new_term not in O and new_term not in border:
                    if origin:
                        border.append((t, new_term))
                    else:
                        border.append(new_term)
        # border = sorted(border, key=lambda t: t.lm())
        return border
    
    def oracle_predict(self, V: List, L: List) -> List:
        """
        Predict the expansion directions using the oracle.
        """
        # find maximal terms in L
        
        # check conditions to use oracle
        min_universe = self.min_universe < len(L)
        relative_gap = len(V)/len(L) > self.relative_gap
        absolute_gap = len(L) - len(V) < self.absolute_gap
        oracle_max_calls = self.oracle_max_calls > self.oracle_calls
        use_oracle = self.use_oracle

        L = find_maximal_terms(L)

        if min_universe and relative_gap and absolute_gap and oracle_max_calls and use_oracle:
            # predict the expansion directions
            predictions = self.oracle.predict(V, L)
            self.oracle_calls += 1

            expansion_lt = [term for _, term in predictions]

            # compute extensions 
            Vpart = [v for v in V if v.lm() in expansion_lt]

            # compute border terms
            border_terms = super().extend_V(Vpart)
        else:
            predictions = []
            border_terms = super().extend_V(V)

        return border_terms
        

    def compute_order_ideal_monomials(self, monomials):
        """
        Compute the order ideal spanned by a given set of monomials.

        Args:
            monomials (list): A list of monomials in a polynomial ring.

        Returns:
            list: The set of all monomials in the order ideal.
        """
        if not monomials:
            return []

        # Get the polynomial ring from the first monomial
        self.ring = monomials[0].parent()

        # Convert monomials to exponent form
        monomial_exponents = [monomial.exponents()[0] for monomial in monomials]

        # Number of variables in the polynomial ring
        num_vars = len(monomial_exponents[0])

        # Compute the order ideal in exponent form
        order_ideal_exponents = set()
        for exponents in monomial_exponents:
            # Generate all divisors by reducing exponents
            divisors = product(*(range(e + 1) for e in exponents))
            order_ideal_exponents.update(divisors)

        # Convert the result back to monomials
        order_ideal_monomials = [self.ring.monomial(*exponents) for exponents in
                                 order_ideal_exponents]

        # Sort the monomials by degree and lexicographical order
        return sorted(order_ideal_monomials, key=lambda t: (t.degree(), t))

    def compute_lstable_span_optimized(self, F: List, L: List,
                                       use_fast_elimination=False, hints: List = None) -> List:
        """
        Optimized version of compute_lstable_span with better handling for large computational universes.
        """
        self.log("Starting optimized L-stable span computation.")

        dataset = [] # list of tuples (L, V, successful_expansion_directions)


        V, _, _ = super().gaussian_elimination([], F,
                                         use_fast_elimination=use_fast_elimination)
        
        # Make V a SortedList
        if self.sorted_V:
            V = SortedList(V)

        while True:
            sequence = [find_maximal_terms(L), V.copy()]
            # Reduce V + \V modulo V

            # compute border terms via oracle
            border_terms = self.oracle_predict(V, L)
            print(len(border_terms))
            self.timings["total_reductions"] = len(border_terms)


            # Gaussian elimination - returns the reduced set of polynomials and the indices of the non-zero reductions
            s = time()
            W, non_zero_reductions_indices, reduction_indices = super().gaussian_elimination(V, border_terms, use_fast_elimination=use_fast_elimination)
            self.timings["gaussian_elimination_times"].append(time()-s)
            W_prime = [f for f in W if f.lm() in L]


            # find all elements in the support of W_prime that are not in L
            L_missing = []

            for f in W_prime:
                for exponents in f.dict().keys():
                    monomial = self.ring.monomial(*exponents)
                    if monomial not in L:
                        L_missing.append(monomial)



            while L_missing:
                # Add missing elements to L
                L.extend(L_missing)
                # Compute the order ideal spanned by L
                L = self.compute_order_ideal_monomials(L)
                # Recompute W_prime
                W_prime = [f for f in W if f.lm() in L]

                # find all elements in the support of W_prime that are not in L
                L_missing = []
                for f in W_prime:
                    for exponents in f.dict().keys():
                        monomial = self.ring.monomial(*exponents)
                        if monomial not in L:
                            L_missing.append(monomial)


            # Compute indices of surviving polynomials in W
            #print()
            surviving_indices = [i for idx, i in enumerate(non_zero_reductions_indices) if W[idx] in W_prime]

            expansion_directions = []
            for idx in surviving_indices:
                poly_idx, var_idx = compute_index_and_expansion_direction(len(self.ring.gens()), idx)
                # convert var_idx to a variable string
                var_string = f"x{var_idx}"
                expansion_directions.append((var_string, poly_idx))
            
            surviving_indices = collect_all_indices(surviving_indices, reduction_indices)

            # collect all indices in the complement of surviving_indices
            non_surviving_indices = [i for i in range(len(border_terms)) if i not in surviving_indices]


            
            non_expansion_polynomials = defaultdict(int)
            for idx in non_surviving_indices:
                poly_idx, var_idx = compute_index_and_expansion_direction(len(self.ring.gens()), idx)
                # convert var_idx to a variable string
                var_string = f"x{var_idx}"
                non_expansion_polynomials[poly_idx] += 1

            # count the number of keys which have value 4
            non_expansion_polynomials = [key for key, value in non_expansion_polynomials.items() if value == len(self.ring.gens())]



            self.efficiency.append((len(surviving_indices), len(border_terms)))

            sequence.append(non_expansion_polynomials)

            sequence.append(expansion_directions)
            
            self.surviving_indices.append(surviving_indices)


            # Update W to the surviving polynomials
            W = W_prime


            if self.verbose:    
                print("V has been extended from", sequence[1])
                print("We added the border terms", sequence[2])
        
            dataset.append(sequence)
            self.datasets.append(dataset)

            if W:
                if self.sorted_V:
                    V.update(W)
                else:
                    V.extend(W)
            else:
                break
            
            

        return V, L

    def compute_border_basis_optimized(self, F: List,
                             use_fast_elimination=False,
                             lstabilization_only=False) -> Tuple[List, List]:
        """
        Implementation of Improved Border Basis algorithm from the paper "Computing Border Bases" by Kehrein & Kreuzer (2005).

        Args:
            F: List of generating polynomials
            use_fast_elimination: Use fast Gaussian elimination
            lstabilization_only: Only compute L-stable span
        Returns:
            Tuple (G, O) where G is the border basis and O is the optimal order ideal
        """
        # Step 1: Compute initial computational universe, order ideal spanned by support of F
        global_start_time = time()

        s = time()
        L = []
        for f in F:
            for exponents in f.dict().keys():
                monomial = self.ring.monomial(*exponents)
                if monomial not in L:
                    L.append(monomial)

        if self.L is not None:
            L = self.L 
        
        # Compute the order ideal spanned by L
        L = self.compute_order_ideal_monomials(L)

        self.timings['step1_initial_order_ideal'] = time() - s

        self.timings['step2_lstable_span_improved'] = []

        self.timings['gaussian_elimination_times'] = []

        while True:
            # Step 2: Compute L-stable span
            s = time()
            M, L = self.compute_lstable_span_optimized(F, L,
                                          use_fast_elimination=use_fast_elimination)
            F = M
            #print("len F", len(F))
            self.timings['step2_lstable_span_improved'].append(time() - s)


            if self.save_universes:
                self.leading_terms.append([f.lm() for f in M])

                # append the universe to the list of universes
                self.universes.append(L)

            if self.verbose:
                print("Universe expanded to size", len(L))

            # Step 3: Check if universe is large enough by checking if border of O is contained in L
            s = time()

            # Construct the set O
            O = L.copy()
            for f in M:
                #print(f.lm())
                if f.lm() in O:
                    O.remove(f.lm())
            
            

            # compute border of O
            border_O = super().border(O)

            # check if the border of O is contained in L
            sufficient_universe = all([m in L for m in border_O])

            # check which border terms are not in L
            if self.save_universes:
                border_terms_missing = [m for m in border_O if m not in L]
                self.border_terms_add.append(border_terms_missing)

            self.timings['step3_check_universe'] = time() - s


            if not sufficient_universe:
                if self.corollarly_23:
                    # every N-th iteration, extend L in all directions
                    if self.count % self.N == 0:
                        L = L + super().border(L)
                        self.count = 0
                    # otherwise, extend L only in the direction of the border of O
                    else:
                        L = self.compute_order_ideal_monomials(L+border_O)
                else:
                    L = L + super().border(L)
                self.count += 1

            else:
                # If the universe is small, disable oracle and continue expanding
                if len(O) >= self.order_ideal_size and self.use_oracle:
                    self.use_oracle = False
                    self.timings["fallback_to_border_basis"] = 1
                    continue
                else:
                    # check if G is a border basis
                    G = self.final_reduction_algorithm(M, O)
                    checker = OBorderBasisChecker(self.ring)
                    is_border_basis = checker.check_oborder_basis(G, O, F)

                    if not is_border_basis:
                        self.timings["fallback_to_border_basis"] = 1
                        self.use_oracle = False
                        continue
                    else:
                        break

        """
        if not lstabilization_only:
            # Employ final reduction algorithm
            s = time()
            O = O # redudant
            
            if self.verbose:
                print(" M", len(M))
            G = self.final_reduction_algorithm(M, O)
            
            if self.verbose:
                print("Size of G", len(G))

        else:
            O = O # redudant
            G = M
        """

                        # sum up all the efficiencies
        non_zero_reductions = sum([e[0] for e in self.efficiency])
        total_expansions = sum([e[1] for e in self.efficiency])

        self.timings['total_efficiency'] = non_zero_reductions/total_expansions

        if self.verbose:
            print("\nExecution times:")
            print("-" * 40)
            for step, t in self.timings.items():
                if step == 'step2_lstable_span_improved':
                    for i, t in enumerate(t):
                        print(f"step2_lstable_span_improved {i}: {t:8.3f} seconds")
                if step == 'gaussian_elimination_times':
                    for i, t in enumerate(t):
                        print(f"gaussian_elimination_times {i}: {t:8.3f} seconds")
                else:
                    print(f"{step:25}: {t:8.3f} seconds")
            print("-" * 40)



            print("Total maximal efficiency gain", non_zero_reductions/total_expansions)

        self.timings["total_time"] = time() - global_start_time

        return G, O, self.timings  

    def final_reduction_algorithm(self, V, O):
        """
        Compute the O-border basis of a zero-dimensional ideal, i.e., transform V into a border basis.

        This is the final reduction algorithm (Proposition 17) from the paper "Computing Border Bases" by Kehrein & Kreuzer (2005).

        Args:
            V (list): A vector basis of the span FL with pairwise different leading terms.
            O (set): Order ideal for border basis.

        Returns:
            list: The O-border basis {g1, ..., gτ}.
        """
        # Initialize VR and create a dictionary for leading terms
        VR = []
        leading_term_map = {}  # Maps leading terms to their polynomials

        # Sort V by leading term
        V = sorted(V, key=lambda v: v.lm())
        
        # Cache for polynomial supports
        support_cache = {}

        while V:
            # Get polynomial with minimal leading term
            v = V.pop(0)
            
            # Compute and cache support if not already cached
            if v not in support_cache:
                support_cache[v] = set(self.ring.monomial(*exponent) for exponent in v.dict().keys())
            
            H = support_cache[v] - ({v.lm()} | set(O))

            if not H:
                # Normalize and store in VR
                normalized_v = v / v.lc()
                VR.append(normalized_v)
                leading_term_map[normalized_v.lm()] = normalized_v
                continue

            # Reduce v using polynomials from VR
            for h in H:
                if h in leading_term_map:
                    wh = leading_term_map[h]
                    ch = v.monomial_coefficient(h) / wh.lc()
                    v -= ch * wh
                else:
                    raise ValueError(f"No wh found in VR for h = {h}")

            # Normalize and store the reduced polynomial
            normalized_v = v / v.lc()
            VR.append(normalized_v)
            leading_term_map[normalized_v.lm()] = normalized_v

        # Construct border basis using the leading term map
        border = super().border(O)
        border_basis = []
        
        for b in border:
            if b in leading_term_map:
                border_basis.append(leading_term_map[b])
            else:
                raise ValueError(f"No polynomial in VR with leading term {b}")

        # Sort the border basis by leading term
        return sorted(border_basis, key=lambda g: g.lm())

    def check_pure_monomials(self, M):
        """
        Check if the leading terms of the polynomials in M contain pure monomials.

        :param M: Set of polynomials
        :return: list of tuples (variable, pure_monomial)
        """
        pure_monomials = []
        for f in M:
            t = f.lm()
            monomial_exponents = t.exponents()[0]
            if all([e == 0 or e == sum(monomial_exponents) for e in monomial_exponents]):
                pure_monomials.append(t)

        return pure_monomials

    def plot_evolution(self):
        """
        Plot the evolution of the computational universe.
        """

        # get the universe terms in exponent form
        universes_exponents = [[tuple(t.exponents()[0]) for t in universe] for universe in self.universes]

        # get the leading terms in exponent form
        leading_terms_exponents = [[tuple(t.exponents()[0]) for t in leading_terms] for leading_terms in self.leading_terms]

        # get the border terms that were added in exponent form
        border_terms_add_exponents = [[tuple(t.exponents()[0]) for t in border_terms] for border_terms in self.border_terms_add]

        # plot the evolution
        print(border_terms_add_exponents)
        plot_multiple_monomials(universes_exponents, leading_terms_exponents, border_terms_add_exponents)



