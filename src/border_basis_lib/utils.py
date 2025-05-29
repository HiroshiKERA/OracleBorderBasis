import matplotlib.pyplot as plt
import math
import itertools as it
from sage.all import *

def corner_terms(border, ring):
    xs = ring.gens()
    corners = []
    for b in border:
        bminus = [b/x for x in xs if x.divides(b)]
        if all([bp not in border for bp in bminus]):
            corners.append(b)

    return corners

def border(O, span_variables=None):
    if span_variables is None:
        span_variables = O[0].args()

    B = []
    for x in span_variables:
        B += [x*o for o in O if x*o not in O]
    return sorted(list(set(B)), key=lambda t: t)

def terms_up_to_degree(d: int, ring):
    """Compute all terms up to degree d."""
    n = ring.ngens()
    terms = [ring(1)]
    for t in range(1, d+1):
        exponents = list(WeightedIntegerVectors(t, [1]*n))
        terms.extend([ring.monomial(*e) for e in exponents])
        
    terms = sorted(terms, key=lambda t: t)
    return terms

def order_from_border(B, ring):
    max_deg = max([b.degree() for b in B])
    terms = terms_up_to_degree(max_deg, ring)
    O = []
    for t in terms:
        if any([t.divides(b) for b in B if t != b]):
            O.append(t)
            
    return O

def subs(F, P):
    '''
    F: list of polynomials
    P: list of points
    '''
    field = P[0, 0].base_ring()
    num_polys  = len(F)
    num_points = P.nrows()
    
    FP = [f(*p) for p, f in it.product(P, F)]

    return MatrixSpace(field, num_points, num_polys)(FP)

def is_regular(M):
    r = min(M.ncols(), M.nrows())
    return M.rank() == r

def keyword_for_numbound(field, bound):
    if field == QQ:
        return {'num_bound': bound}
    if field == ZZ:
        return {'x': -bound, 'y': bound}
    if field == RR:
        return {'min': -bound, 'max': bound}
    if field.is_finite():
        return {}
    

def is_all_divisors_in(t: Polynomial, O) -> bool:
    """
    Check if adding term t to order ideal O maintains the order ideal property.
    
    Since O is already an order ideal, we only need to check immediate divisors
    (reducing degree by 1 for each variable).
    """
    if t.degree() in (0, 1): 
        return True
    
    return all([xi in O and t/xi in O for xi in t.args() if xi.divides(t)])


def is_order_ideal(O):
    max_deg = max([o.degree() for o in O])
    
    if 1 not in O:
        return False
    
    O_ = [1]
    for d in range(1, max_deg+1):
        terms = [o for o in O if o.degree() == d]
        for t in terms:
            if not is_all_divisors_in(t, O_):
                return False
        
        O_ += terms
    
    return True

def compute_stairs(L):
    """
    Computes the stairs boundary for a given set of points in L.

    Parameters:
        L (list of tuple): List of points representing monomials in (x, y) coordinates.

    Returns:
        stairs_x, stairs_y (list): Coordinates of the stairs path.
    """
    # Extract x and y coordinates
    x_coords, y_coords = zip(*L)

    # Find the starting point: highest y-coordinate at x=0
    start_point = max([p for p in L if p[0] == 0], key=lambda p: p[1])
    current_point = start_point

    stairs_path = [current_point]

    while current_point != (max(x_coords), 0):  # Continue until the bottom-right corner
        x, y = current_point

        # Check if moving right is possible
        right_point = (x + 1, y)
        if right_point in L:
            current_point = right_point
        else:
            # Otherwise, move down
            down_point = (x, y - 1)
            if down_point in L:
                current_point = down_point
            else:
                # If both directions are blocked, break the loop
                break

        stairs_path.append(current_point)

    # Add the final point to close the boundary at the bottom-right
    stairs_path.append((max(x_coords), 0))

    # Extract x and y coordinates for stairs path
    stairs_x, stairs_y = zip(*stairs_path)

    # Adjust the stairs to make it visually appealing
    stairs_y = [x + 0.25 for x in stairs_y]
    stairs_y[-1] -= 0.25
    stairs_x = [x + 0.25 for x in stairs_x]
    stairs_y = [stairs_y[0]] + stairs_y
    stairs_x = [0] + stairs_x

    return stairs_x, stairs_y


def plot_multiple_monomials(L_list, leading_terms_list, added_terms_list):
    """
    Plots multiple sets of monomials in a grid with up to 4 plots per row.

    Parameters:
        L_list (list of list of tuple): List of sets of points representing monomials in (x, y) coordinates.
        leading_terms_list (list of list of tuple): List of leading terms corresponding to each set in L_list.
        added_terms_list (list of list of tuple): List of added (border missing) terms corresponding to each set.
    """
    num_plots = len(L_list)
    num_cols = 4  # Maximum 4 plots per row
    num_rows = math.ceil(num_plots / num_cols)  # Number of rows needed

    # Determine global axis limits for consistent scaling
    all_x = [x for L in L_list for x, y in L]
    all_y = [y for L in L_list for x, y in L]
    x_min, x_max = 0, max(all_x) + 1
    y_min, y_max = 0, max(all_y) + 1

    plt.figure(figsize=(16, 4 * num_rows))

    for idx, (L, leading_terms, added_terms) in enumerate(zip(L_list, leading_terms_list, added_terms_list)):
        # Determine subplot position
        row, col = divmod(idx, num_cols)
        plt.subplot(num_rows, num_cols, idx + 1)

        # Compute stairs for the current set
        stairs_x, stairs_y = compute_stairs(L)

        # Extract x and y coordinates for plotting
        x_coords, y_coords = zip(*L)

        # Plot points in L
        plt.scatter(x_coords, y_coords, color='blue', label='Monomials in L')

        # Highlight leading terms
        for x, y in leading_terms:
            plt.scatter(x, y, color='red', label='Leading Terms in V')

        # Highlight added (border missing) terms
        for x, y in added_terms:
            plt.scatter(x, y, color='green', label='Missing Border Terms')

        # Draw the stairs boundary
        plt.plot(stairs_x, stairs_y, color='purple', linewidth=2, label='Boundary')

        # Set plot title and integer axis ticks
        plt.title(f'Universe {idx + 1}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks(range(x_min, x_max))
        plt.yticks(range(y_min, y_max))
        plt.xlim(x_min-0.25, x_max)
        plt.ylim(y_min-0.25, y_max)
        plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

    # Add empty subplots for unused slots
    for empty_idx in range(idx + 1, num_rows * num_cols):
        plt.subplot(num_rows, num_cols, empty_idx + 1)
        plt.axis('off')  # Turn off unused subplot

    plt.tight_layout()
    plt.show()

def find_maximal_terms(terms):
    """
    Returns terms that are not divisors of any other term in the set.
    
    For example, with terms {x, y, xy}, returns {xy} since xy doesn't divide
    any other term in the set.
    
    Parameters:
        terms: A list of polynomial terms
        
    Returns:
        A list of terms that don't divide any other term in the set
    """
    result = []
    for t in terms:
        # Check if t divides any other term in the set
        if not any(t != s and t.divides(s) for s in terms):
            result.append(t)
    
    return result

def collect_all_indices(surviving_indices, reduction_dict):
    """
    Collect all unique indices from the reduction dictionary based on the surviving indices.

    This function takes a list of surviving indices and a reduction dictionary, and returns a sorted list of all unique
    indices that can be reached through the reduction dictionary. The reduction dictionary maps indices to lists of 
    other indices that should be included.

    Args:
        surviving_indices (list of int): A list of initial indices to start the collection from.
        reduction_dict (dict of int: list of int): A dictionary where keys are indices and values are lists of indices 
                                                   that should be included if the key is included.

    Returns:
        list of int: A sorted list of all unique indices collected from the reduction dictionary.

    Example:
        surviving_indices = [22, 101, 102]
        reduction_dict = {
            22: [7, 6],
            7: [3, 2, 1],
            101: [50, 51],
            50: [25],
            25: [12, 13]
        }
        result = collect_all_indices(surviving_indices, reduction_dict)
        # result will be [1, 2, 3, 6, 7, 12, 13, 22, 25, 50, 51, 101, 102]
    """
    result = set()  # Using a set to avoid duplicates
    to_process = list(surviving_indices)  # Create a queue of numbers to process
    
    while to_process:
        current = to_process.pop()
        if current not in result:  # If we haven't processed this number yet
            result.add(current)
            # Add all numbers from this key's reduction list to process
            if current in reduction_dict:
                to_process.extend(reduction_dict[current])
    
    return sorted(list(result))  # Convert back to sorted list

def compute_index_and_expansion_direction(num_vars, idx):
    """
    Compute the index and expansion direction for a given index in the reduction dictionary.
    """
    # Compute the index of the expansion direction
    poly_idx = idx // num_vars
    var_idx = idx % num_vars

    return poly_idx, var_idx

def expand_within_universe(V, universe):
    """
    Enlarge the given set of polynomials V to a vector space F by adding for each basis element v of V all
    products tv such that t ∈ T and LT(tv) ∈ U. 
    """
    # Get the polynomial ring from the first polynomial in V
    ring = V[0].parent()
    
    # Initialize the set of new polynomials to add
    new_polynomials = set()
    
    # Iterate over each polynomial in V
    for v in V:
        # Iterate over each monomial in the universe
        for t in universe:
            # Compute the product of the monomial and the polynomial
            tv = t * v
            # Check if the leading term of the product is in the universe
            if tv.lt() in universe:
                new_polynomials.add(tv)
    
    # Combine the original set V with the new polynomials
    extended_basis = V + list(new_polynomials)
    
    return extended_basis
    
    

    
    