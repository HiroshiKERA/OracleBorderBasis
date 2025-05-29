import unittest
import sys
import json
import os
from sage.all import PolynomialRing, QQ
#from scipy.special import order

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import border_basis


class TestGaussianElimination(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up paths for the test instances directory."""
        cls.instances_dir = 'tests/instances'

    def load_instance(self, filename):
        """Load a JSON instance file and parse polynomials into SageMath objects."""
        with open(filename, 'r') as f:
            data = json.load(f)

        # Create polynomial ring with specified variables
        variables = data["variables"]
        R = PolynomialRing(QQ, variables, order="degrevlex")

        # Parse V, F, and expected_output as Sage polynomials
        V = [R(poly_str) for poly_str in data["V"]]
        F = [R(poly_str) for poly_str in data["F"]]
        expected_output = [R(poly_str) for poly_str in data["expected_output"]]

        return V, F, expected_output, R

    def test_gaussian_elimination_instances(self):
        """Test gaussian_elimination on all instances in the directory."""

        for instance_file in os.listdir(self.instances_dir):
            if instance_file.endswith('.json'):
                # Load and parse instance data
                filepath = os.path.join(self.instances_dir, instance_file)
                V, F, expected_output, R = self.load_instance(filepath)

                # sort the polynomials in V by degree
                V = sorted(V, key=lambda p: p.lt())

                calculator = border_basis.BorderBasisCalculator(R)


                # Run fast gaussian_elimination
                result = calculator.gaussian_elimination_fast(V, F)

                # sort the result by degree
                result = sorted(result, key=lambda p: p.lt())

                # Verify the result
                self.assertEqual(result, expected_output,
                                 f"Failed for instance {instance_file}")


# Run the tests
if __name__ == '__main__':
    unittest.main()