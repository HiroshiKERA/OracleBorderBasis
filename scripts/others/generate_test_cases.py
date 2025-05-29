import json
import os
from sage.all import PolynomialRing, QQ
#from scipy.special import order

import border_basis


def generate_expected_output(instance_file):
    """Generate and save expected output for a given JSON instance file."""
    with open(instance_file, 'r') as f:
        data = json.load(f)

    # Create the polynomial ring with specified variables
    variables = data["variables"]
    R = PolynomialRing(QQ, variables, order="degrevlex")

    # Parse V and F into SageMath polynomials
    V = [R(poly_str) for poly_str in data["V"]]
    F = [R(poly_str) for poly_str in data["F"]]


    # Run gaussian_elimination to get the expected output
    calculator = border_basis.BorderBasisCalculator(R)

    # Note this sorts the polynomials in V by degree
    V = sorted(V, key=lambda p: p.lt())

    expected_output = calculator.gaussian_elimination(V, F)

    # Convert expected output polynomials back to strings for JSON serialization
    expected_output = sorted(expected_output, key=lambda p: p.lt())
    data["expected_output"] = [str(poly) for poly in expected_output]

    # Save updated data back to the JSON file
    with open(instance_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Expected output generated and saved for {instance_file}")

def generate_M_expected_output(instance_file):
    """Generate and save expected output for M of stable span computation for a given JSON instance file."""
    with open(instance_file, 'r') as f:
        data = json.load(f)

    # Create the polynomial ring with specified variables
    variables = data["variables"]
    R = PolynomialRing(QQ, variables, order="degrevlex")

    # F into SageMath polynomials
    F = [R(poly_str) for poly_str in data["F"]]

    # Run stable_span to get the expected output
    calculator = border_basis.BorderBasisCalculator(R)

    # Define a set of weights
    d = max(f.degree() for f in F)
    terms = calculator.terms_up_to_degree(d)
    weights = {t: 1 for i, t in enumerate(terms)}

    # Compute the border basis, no fast elimination and lstabilization_only
    G, O, _ = calculator.compute_border_basis(F, weights,
                                              use_fast_elimination=False,
                                              lstabilization_only=True)

    # Convert expected output polynomials back to strings for JSON serialization
    expected_output = sorted(G, key=lambda p: p.lt())
    data["M"] = [str(poly) for poly in expected_output]

    # Save updated data back to the JSON file
    with open(instance_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Expected output M generated and saved for {instance_file}")


# Directory containing test instances
instances_dir = 'tests/instances'

# Generate expected output for all instances
for instance_file in os.listdir(instances_dir):
    if instance_file.endswith('.json'):
        generate_expected_output(os.path.join(instances_dir, instance_file))

# For lstable_span_test
instances_dir = 'tests/stable_span_instances'

for instance_file in os.listdir(instances_dir):
    if instance_file.endswith('.json'):
        generate_M_expected_output(os.path.join(instances_dir, instance_file))