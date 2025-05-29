import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from sage.all import PolynomialRing, QQ
import border_basis


class StableSpanBenchmark:
    def __init__(self, instances_dir):
        """
        Initialize the benchmark class with the directory of JSON instances.
        """
        self.instances_dir = instances_dir
        self.fast_times = []
        self.normal_times = []

    def load_instance(self, filepath):
        """
        Load a JSON instance file and parse polynomials into SageMath objects.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Create polynomial ring with specified variables
        variables = data["variables"]
        R = PolynomialRing(QQ, variables.split(', '), order="degrevlex")

        # Parse F as Sage polynomials
        F = [R(poly_str) for poly_str in data["F"]]
        return F, R

    def benchmark_instance(self, filepath):
        """
        Benchmark \( M \)-stable span computation with and without fast Gaussian elimination.
        """
        F, R = self.load_instance(filepath)
        calculator = border_basis.BorderBasisCalculator(R)

        # Determine the maximum degree for stable span computation
        d = max(f.degree() for f in F)

        # Define terms and weights
        terms = calculator.terms_up_to_degree(d)
        weights = {t: 1 for t in terms}

        # Benchmark with fast Gaussian elimination
        start_time = time.time()
        calculator.compute_border_basis(F, weights, use_fast_elimination=True, lstabilization_only=True)
        fast_elapsed = time.time() - start_time

        # Benchmark without fast Gaussian elimination
        start_time = time.time()
        calculator.compute_border_basis(F, weights, use_fast_elimination=False, lstabilization_only=True)
        normal_elapsed = time.time() - start_time

        return fast_elapsed, normal_elapsed

    def run_benchmark(self):
        """
        Run the benchmark for all instances in the directory.
        """
        for instance_file in os.listdir(self.instances_dir):
            if instance_file.endswith('.json'):
                filepath = os.path.join(self.instances_dir, instance_file)
                fast_time, normal_time = self.benchmark_instance(filepath)
                self.fast_times.append(fast_time)
                self.normal_times.append(normal_time)

    def plot_results(self):
        """
        Plot a log-log scatter plot comparing fast and normal runtimes.
        """
        # Ensure that times have been benchmarked
        if not self.fast_times or not self.normal_times:
            raise ValueError("No benchmark data available. Run `run_benchmark()` first.")

        log_fast = np.log10(self.fast_times)
        log_normal = np.log10(self.normal_times)

        plt.figure(figsize=(8, 8))
        plt.scatter(log_fast, log_normal, alpha=0.7, label="Instance Runtime")

        # Add a reference line (y = x) for comparison
        max_val = max(max(log_fast), max(log_normal))
        min_val = min(min(log_fast), min(log_normal))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')

        # Add a line showing 10x speedup (y = x - 1)
        plt.plot([min_val, max_val], [min_val + 1, max_val + 1], color='blue',
                 linestyle='-', label='10x Speedup Line')

        plt.title("Log-Log Scatter Plot of L-Stable Span Computation Times for fast GE and normal GE")
        plt.xlabel("Log10(Fast GE Computation Time)")
        plt.ylabel("Log10(Normal GE Computation Time)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


# Main script
if __name__ == '__main__':
    # Directory containing instances
    instances_dir = 'tests/stable_span_instances'

    # Instantiate the benchmark class and run the benchmark
    benchmark = StableSpanBenchmark(instances_dir)
    benchmark.run_benchmark()

    # Plot the results
    benchmark.plot_results()