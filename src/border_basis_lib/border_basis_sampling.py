from sage.all import *
from sage.combinat.partition import Partitions
import itertools as it
from random import shuffle, choice
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

from .utils import border, subs, is_regular, keyword_for_numbound

@dataclass
class Segment:
    """Represents a segment in n-dimensional space defined by two endpoints and an axis"""
    endpoints: List        # List[np.ndarray], two endpoints (n-dim vector each)
    
    def __post_init__(self):
        """Initialize bounds and dimension after instantiation"""
        self.lb = np.array(np.minimum.reduce(self.endpoints), dtype=int)
        self.ub = np.array(np.maximum.reduce(self.endpoints), dtype=int)
        self.n = len(self.lb)
    
    def __repr__(self):
        return f'Segment(lb={self.lb}, ub={self.ub}, endpoint={self.endpoints})'

    def __hash__(self):
        return hash((tuple(self.lb), tuple(self.ub)))

@dataclass
class NeighborSegments:
    """Collection of segments that intersect at a point"""
    segments: List[Segment]
    intersecting_point: np.ndarray
    
    def __post_init__(self):
        """Initialize dimension, max point and validity after instantiation"""
        self.n = self.segments[0].n
        max_point = np.vstack([segment.ub for segment in self.segments])
        self.max_point = np.min(max_point + np.eye(self.n, dtype=int) * 100000, axis=0)
        # self.valid = not np.all((self.max_point - self.intersecting_point) <= 1)
        self.valid = np.sum(self.max_point != self.intersecting_point) > 1 and not np.all(self.max_point - self.intersecting_point <= 1)
    
    def sampling(self) -> np.ndarray:
        """Sample a point between intersecting_point and max_point"""
        # print(f'Sampling between {self.intersecting_point} and {self.max_point}')
        nondegenerated = np.array(self.intersecting_point < self.max_point + 1)
        point = self.intersecting_point.copy()
        point[nondegenerated] = np.random.randint(self.intersecting_point[nondegenerated], self.max_point[nondegenerated] + 1)  
        return point
    
    def split_at(self, splitpoint: np.ndarray) -> List['NeighborSegments']:
        """Split segments at given point to create new NeighborSegments"""
        
        new_segment = Segment(deepcopy([self.intersecting_point, splitpoint]))
        
        new_neighborsegments = []
        for i in range(self.n):
            new_segments = []
            for j, segment in enumerate(self.segments):
                if i != j:
                    lb = segment.lb
                    lb[j] = splitpoint[j]
                    ub = segment.ub
                    new_segments.append(Segment([lb, ub]))
                else:
                    new_segments.append(deepcopy(new_segment))
                    
            new_intersecting_point = (self.intersecting_point + 
                                    np.eye(self.n)[i] * (splitpoint[i] - self.intersecting_point[i]))
            new_neighborsegment = NeighborSegments(deepcopy(new_segments), 
                                                 new_intersecting_point)
            new_neighborsegments.append(new_neighborsegment)
        return new_neighborsegments
    

class BorderBasisGenerator:
    def __init__(self, ring): 
        """
        Initialize generator for computing border bases
        
        Parameters:
        -----------
        ring : Ring
            Ring object that defines the polynomial ring and number of variables
        """
        self.ring = ring 
        self.n = ring.ngens()
    
    def sample_order_ideal(self, degree_bounds: List[int], 
                   max_iters: int = 100, track_changes=False) -> List[Tuple[int, ...]]:
        """
        Generates border by iteratively splitting segments and sampling points
        
        Parameters:
        -----------
        degree_bounds : List[int]
            Maximum degree for each variable. Length must equal self.n
        max_iters : int
            Maximum number of iterations for segment splitting
        track_changes : bool
            Whether to track changes in the order ideal (for debugging or demo)
            
        Returns:
        --------
        List[Tuple[int, ...]]
            List of points forming the order ideal, sorted lexicographically
        """
        
        origin = np.zeros(self.n, dtype=int)
        max_point = np.array(degree_bounds)
        
        S = []
        for i in range(self.n):
            endpoint = max_point.copy()
            endpoint[i]= 0
            segment = Segment([origin, endpoint])
            S.append(segment)
            
        N = [NeighborSegments(deepcopy(S), origin)]
        T = []
        O = []
        Os = []
        
        O_axis = []
        for i in range(self.n):
            canonical_basis = np.eye(self.n,  dtype=int)[i]
            O_axis.extend(self.hypercube_points(origin, canonical_basis * degree_bounds[i], exclude_max=False))

        for i in range(max_iters):
            if i == max_iters - 1:
                print(f'In sample_order_ideal: Loop out after {max_iters} iterations.')  # not failure, just a warning
                break
            if not N:
                break
            
            neighbor_segment = N.pop()
            new_neighborsegments = []
            
            splitpoint = neighbor_segment.sampling()
            
            for new_neighborsegment in neighbor_segment.split_at(splitpoint):
                if new_neighborsegment.valid:
                    new_neighborsegments.append(new_neighborsegment)
            
            neighbor_segment.max_point = splitpoint
            T.append(neighbor_segment)
            
            if track_changes:
                # print(self.span_order_ideal(T[-1:]))
                current_order_ideal = deepcopy(O_axis + self.span_order_ideal(T))
                current_order_ideal = list(set(current_order_ideal))
                current_order_ideal.sort(key=lambda x: (-sum(x), *reversed(x)))
                if len(Os) == 0 or Os[-1] != current_order_ideal:
                    Os.append(current_order_ideal)
                    
            N.extend(new_neighborsegments)
            shuffle(N)

        O = list(set(O_axis + self.span_order_ideal(T)))
        
        # grevlex
        O.sort(key=lambda x: (-sum(x), *reversed(x)))
        
        if track_changes:
            return O, Os
        else:
            return O
        
    def span_order_ideal(self, neighbor_segments: List[NeighborSegments]):
        order_ideal = []
        for neighbor_segment in neighbor_segments:
            u = neighbor_segment.intersecting_point
            v = neighbor_segment.max_point
            points = self.hypercube_points(u, v, exclude_max=False)
            order_ideal.extend(deepcopy(points))
        
        return order_ideal
            

    def hypercube_points(self, u: np.ndarray, v: np.ndarray, 
                        exclude_max: bool = True) -> List[Tuple[int, ...]]:
        """
        Generate all integer points in hypercube [u,v]
        
        Parameters:
        -----------
        u, v : np.ndarray
            Lower and upper bounds defining the hypercube
        exclude_max : bool
            Whether to exclude the maximum point v
            
        Returns:
        --------
        List[Tuple[int, ...]]
            All grid points in hypercube, excluding v if specified
        """
        u = np.minimum(u, v)
        v = np.maximum(u, v)
        n = len(u)
        grid_ranges = [np.arange(u[i], v[i] + 1, dtype=int) for i in range(n)]
        points = [tuple(map(int, p)) for p in it.product(*grid_ranges) 
                 if not (exclude_max and np.array_equal(p, v))]
        return points

    def random_order_ideal(self, 
                           degree_bounds: List[int], 
                           total_degree_bound: int = None,
                           degree_lower_bounds: List[int] = None, 
                           track_changes=False) -> List[Tuple[int, ...]]:
        """
        Generate a random order ideal with random bounds up to given maximum
        
        Parameters:
        -----------
        degree_bounds : List[int]
            Maximum possible degree for each variable
        # **kwargs : 
        #     Additional arguments passed to find_border
        
        Returns:
        --------
        List[Tuple[int, ...]]
            Randomly generated order ideal
        """
        if degree_lower_bounds is None:
            degree_lower_bounds = 0
            
        if total_degree_bound is None:
            upper_bounds = np.random.randint(degree_lower_bounds, np.array(degree_bounds)+1)
        else:
            random_total_degree = np.random.randint(1, total_degree_bound+1)
            partitions = list(Partitions(random_total_degree, max_length=self.n, max_part=max(degree_bounds)))
            partition = choice(partitions)
            partition = partition + [0] * (self.n - len(partition))
            shuffle(partition)
            
            upper_bounds = partition
            
            # upper_bounds = np.random.randint(degree_lower_bounds, total_degree_bound+1)
            
        return self.sample_order_ideal(upper_bounds, track_changes=track_changes)
    
    def compute_border_basis(self, B: List[Any], O: List[Any], P: Any) -> Tuple[Optional[List], bool]:
        """
        Find a border basis for given border B and order ideal O at points P
        
        Parameters:
        -----------
        B : List[Any]
            List of border terms (as ring elements)
        O : List[Any]
            List of order ideal terms (as ring elements)
        P : Any
            Matrix of evaluation points
        
        Returns:
        --------
        Tuple[Optional[List], bool]
            - First element is the basis coefficients if found, None if not found
            - Second element is True if basis was found successfully
        """
        ring = self.ring
        O = [ring(o) for o in O]
        OP = subs(O, P)
        
        if not is_regular(OP):
            # print('The matrix is not regular.')
            return None, False
        
        B = [ring(b) for b in B]
        BP = subs(B, P)
        M = BP.augment(OP)
        V = M.transpose().kernel().basis()

        return V, True
    
    def random_border_basis(self, degree_bounds: List[int], 
                            max_sampling: int = 100,
                            total_degree_bound: int = None,
                            degree_lower_bounds: Optional[List[int]] = None, 
                            **kwargs) -> Dict[str, Any]:
        """
        Generate a random border basis with given degree bounds
        
        Parameters:
        -----------
        degree_bounds : List[int]
            Maximum degree for each variable
        max_sampling: int
            Maximum number of attempts to find valid evaluation points
        total_degree_bound: int
            Maximum total degree of the border basis
        degree_lower_bound : Optional[List[int]]
            Minimum degree for each variable (defaults to [0,...,0])
        **kwargs : 
            Additional arguments passed to find_border
        
        Returns:
        --------
        Dict[str, Any] with keys:
            - basis: List of basis elements
            - order_coeff: Coefficients of the border basis
            - border: Border terms
            - order: Order ideal terms
            - points: Evaluation points
            - success: Whether computation succeeded
        """
        assert len(degree_bounds) == self.n
        
        if degree_lower_bounds is None:
            degree_lower_bounds = [0] * self.n
        
        # Generate random border and convert to polynomial ring
        ring = self.ring
        O = self.random_order_ideal(np.array(degree_bounds)-1, 
                                    total_degree_bound=total_degree_bound-1,
                                    degree_lower_bounds=degree_lower_bounds)
        # plot_border(degree_bounds, O)
        O = [ring.monomial(*o) for o in O]
        B = border(O)
        B_exponents = [t.exponents()[0] for t in B]
        B_exponents.sort(key=lambda x: (-sum(x), *reversed(x)))
        B = [ring.monomial(*e) for e in B_exponents]
        
        # Find basis by sampling points
        MSpace = MatrixSpace(ring.base_ring(), len(O), self.n)
        success = False
        
        samlping_retrials = 0
        for i in range(max_sampling):
            if i == max_sampling - 1:
                print(f'Failed to find a border basis after {max_sampling} sampling of points.')
                break
                
            P = MSpace.random_element(**keyword_for_numbound(ring.base_ring(), 10))
            V, success = self.compute_border_basis(B, O, P)

            
            if success:
                samlping_retrials = i
                break
            
        
        # Construct border basis
        G = []
        if success:
            G = Matrix(B + O) * Matrix(V).T
            G = G[0]
            G = [ring(g) for g in G]
        
        return {
            'basis': G,
            'order_coeff': V, 
            'border': B,
            'order': O,
            'points': P,
            'sampling_retrials': samlping_retrials,
            'success': success
        }