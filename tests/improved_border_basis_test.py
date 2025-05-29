import unittest
import os
import json
from sage.all import PolynomialRing, QQ
from pprint import pprint
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# Replace this with the actual import for ImprovedBorderBasisCalculator
import improved_border_basis as ibb


class TestImprovedBorderBasis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up paths for the test instances directory."""
        cls.instances_dir = 'tests/improved_border_basis_instances'

    def load_instance(self, filename):
        """Load a JSON instance file and parse polynomials into SageMath objects."""
        with open(filename, 'r') as f:
            data = json.load(f)

        # Create polynomial ring with specified variables
        variables = data["variables"]
        R = PolynomialRing(QQ, variables.split(', '), order="degrevlex")

        # Parse input and output polynomials as Sage polynomials
        input_polynomials = [R(poly_str) for poly_str in
                             data["F"]]
        expected_output = [R(poly_str) for poly_str in
                           data["border_basis"]]

        return input_polynomials, expected_output, R

    def compute_output(self, F, R, use_fast_elimination=False,
                       lstabilization_only=True):
        """Compute the output using ImprovedBorderBasisCalculator."""
        calculator = ibb.ImprovedBorderBasisCalculator(R)

        # Compute the border basis
        G, O, _ = calculator.compute_border_basis_optimized(
            F, use_fast_elimination=use_fast_elimination,
            lstabilization_only=lstabilization_only
        )

        # Perform the final reduction algorithm
        output = calculator.final_reduction_algorithm(G, O)

        # Return the sorted output for comparison
        return sorted(output, key=lambda p: p.lt())

    def test_improved_border_basis_instances(self):
        """Test ImprovedBorderBasisCalculator on all instances in the directory."""
        for instance_file in os.listdir(self.instances_dir):
            if instance_file.endswith('.json'):
                filepath = os.path.join(self.instances_dir, instance_file)

                # Load and parse instance data
                F, expected_output, R = self.load_instance(filepath)

                # Compute the output
                result = self.compute_output(F, R)

                # Compare the result with the expected output
                self.assertEqual(result, expected_output,
                                 f"Failed for instance {instance_file}: {pprint(result)}")


# Run the tests
if __name__ == '__main__':
    unittest.main()