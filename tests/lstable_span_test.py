import unittest
import sys
import json
import os
from sage.all import PolynomialRing, QQ

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import border_basis


class TestStableSpanComputation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up paths for the test instances directory."""
        cls.instances_dir = 'tests/stable_span_instances'

    def load_instance(self, filename):
        """Load a JSON instance file and parse polynomials into SageMath objects."""
        with open(filename, 'r') as f:
            data = json.load(f)

        # Create polynomial ring with specified variables
        variables = data["variables"]
        R = PolynomialRing(QQ, variables.split(', '), order="degrevlex")

        # Parse F as Sage polynomials
        F = [R(poly_str) for poly_str in data["F"]]
        expected_output = [R(poly_str) for poly_str in data.get("M", [])]

        return F, expected_output, R

    def generate_expected_output(self, F, R):
        """Generate the expected output for M-stable span computation."""
        calculator = border_basis.BorderBasisCalculator(R)

        # Determine the maximum degree for stable span computation
        d = max(f.degree() for f in F)

        # Define terms and weights
        terms = calculator.terms_up_to_degree(d)
        weights = {t: 1 for t in terms}

        # Compute the border basis with lstabilization_only=True and fast elimination
        G, _, _ = calculator.compute_border_basis(F, weights,
                                                  use_fast_elimination=True,
                                                  lstabilization_only=True)

        # Sort the output by degree and return as strings
        return sorted(G, key=lambda p: p.lt())


    def test_stable_span_computation_instances(self):
        """Test M-stable span computation on all instances in the directory."""
        for instance_file in os.listdir(self.instances_dir):
            if instance_file.endswith('.json'):
                # Load and parse instance data
                filepath = os.path.join(self.instances_dir, instance_file)
                F, expected_output, R = self.load_instance(filepath)

                # Compute the stable span
                result = self.generate_expected_output(F, R)

                # Compare the result with the expected output
                self.assertEqual(result, expected_output,
                                     f"Failed for instance {instance_file}")


# Run the tests
if __name__ == '__main__':
    unittest.main()