#!/usr/bin/env python3
"""
Test suite for EoH-PCTSP framework
"""

import unittest
import numpy as np
import os
import tempfile
import shutil
from PCTSP import PCTSPData, PCTSPSolution, construct_initial_solution, evaluate_operator, load_instances
from pctsp_prompts import GetPCTSPPrompts

class TestPCTSPBasics(unittest.TestCase):
    """Test basic PCTSP functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Create a simple test instance
        self.test_instance = PCTSPData(
            instance_id=1,
            size=5,
            depot=np.array([0.0, 0.0]),
            locations=np.array([
                [1.0, 0.0],  # Node 0
                [0.0, 1.0],  # Node 1
                [2.0, 0.0],  # Node 2
                [0.0, 2.0],  # Node 3
                [1.0, 1.0]   # Node 4
            ]),
            penalties=np.array([0.5, 0.3, 0.8, 0.4, 0.6]),
            prizes=np.array([0.3, 0.4, 0.2, 0.5, 0.3])
        )
        
        # Set global DATA
        import PCTSP
        PCTSP.DATA = self.test_instance
    
    def test_pctsp_data_creation(self):
        """Test PCTSPData creation"""
        self.assertEqual(self.test_instance.size, 5)
        self.assertEqual(self.test_instance.total_prize, 1.0)
        self.assertEqual(len(self.test_instance.locations), 5)
        self.assertEqual(len(self.test_instance.penalties), 5)
        self.assertEqual(len(self.test_instance.prizes), 5)
    
    def test_solution_creation(self):
        """Test PCTSPSolution creation"""
        tour = [0, 1, 3]  # Visit nodes 0, 1, 3
        solution = PCTSPSolution(tour)
        
        self.assertEqual(solution.tour, [0, 1, 3])
        self.assertEqual(set(solution.unvisited), {2, 4})
        self.assertGreater(solution.objective(), 0)
        
        # Check prize calculation
        expected_prize = self.test_instance.prizes[0] + self.test_instance.prizes[1] + self.test_instance.prizes[3]
        self.assertAlmostEqual(solution.total_prize(), expected_prize, places=6)
    
    def test_solution_feasibility(self):
        """Test solution feasibility checking"""
        # Create a solution with enough prize
        tour = [1, 3, 4]  # Total prize = 0.4 + 0.5 + 0.3 = 1.2 >= 1.0
        solution = PCTSPSolution(tour)
        self.assertTrue(solution.is_feasible())
        
        # Create a solution with insufficient prize
        tour = [0]  # Total prize = 0.3 < 1.0
        solution = PCTSPSolution(tour)
        self.assertFalse(solution.is_feasible())
    
    def test_solution_operations(self):
        """Test solution insert/remove operations"""
        solution = PCTSPSolution([0, 1])
        
        # Test insert
        solution.insert(2, 1)  # Insert node 2 at position 1
        self.assertEqual(solution.tour, [0, 2, 1])
        self.assertNotIn(2, solution.unvisited)
        
        # Test remove
        solution.remove(2)
        self.assertEqual(solution.tour, [0, 1])
        self.assertIn(2, solution.unvisited)
    
    def test_initial_solution_construction(self):
        """Test initial solution construction"""
        solution = construct_initial_solution(use_greedy=True)
        
        self.assertIsInstance(solution, PCTSPSolution)
        self.assertTrue(solution.is_feasible())
        self.assertGreater(len(solution.tour), 0)
        self.assertGreaterEqual(solution.total_prize(), self.test_instance.total_prize)

class TestPCTSPPrompts(unittest.TestCase):
    """Test PCTSP prompts"""
    
    def test_prompts_creation(self):
        """Test prompt creation"""
        prompts = GetPCTSPPrompts()
        
        self.assertIsNotNone(prompts.get_task_description())
        self.assertIn("PCTSP", prompts.get_task_description())
        self.assertIn("llm_repair", prompts.get_task_description())
        
        inputs = prompts.get_function_inputs()
        outputs = prompts.get_function_outputs()
        
        self.assertIsInstance(inputs, list)
        self.assertIsInstance(outputs, list)
        self.assertGreater(len(inputs), 0)
        self.assertGreater(len(outputs), 0)

class TestPCTSPOperators(unittest.TestCase):
    """Test PCTSP operators"""
    
    def setUp(self):
        """Set up test data"""
        # Create test instance
        self.test_instance = PCTSPData(
            instance_id=1,
            size=4,
            depot=np.array([0.0, 0.0]),
            locations=np.array([
                [1.0, 0.0],  # Node 0
                [0.0, 1.0],  # Node 1
                [2.0, 0.0],  # Node 2
                [0.0, 2.0]   # Node 3
            ]),
            penalties=np.array([0.5, 0.3, 0.8, 0.4]),
            prizes=np.array([0.3, 0.4, 0.2, 0.5])
        )
        
        # Set global DATA
        import PCTSP
        PCTSP.DATA = self.test_instance
    
    def test_destroy_operators(self):
        """Test destroy operators"""
        from PCTSP import random_removal, worst_removal
        import numpy.random as rnd
        
        # Create initial solution
        solution = PCTSPSolution([0, 1, 3])
        original_tour_length = len(solution.tour)
        
        rng = rnd.default_rng(2345)
        
        # Test random removal
        damaged = random_removal(solution, rng)
        self.assertLessEqual(len(damaged.tour), original_tour_length)
        
        # Test worst removal
        solution2 = PCTSPSolution([0, 1, 3])
        damaged2 = worst_removal(solution2, rng)
        self.assertLessEqual(len(damaged2.tour), original_tour_length)
    
    def test_repair_operators(self):
        """Test repair operators"""
        from PCTSP import greedy_repair
        import numpy.random as rnd
        
        # Create damaged solution
        solution = PCTSPSolution([0])  # Only one node, not feasible
        self.assertFalse(solution.is_feasible())
        
        rng = rnd.default_rng(2345)
        
        # Test greedy repair
        repaired = greedy_repair(solution, rng)
        self.assertTrue(repaired.is_feasible())
        self.assertGreaterEqual(repaired.total_prize(), self.test_instance.total_prize)
    
    def test_operator_evaluation(self):
        """Test operator evaluation"""
        # Create a simple repair operator
        def simple_repair(state, rng, **kwargs):
            # Insert all unvisited nodes
            while state.unvisited:
                node = state.unvisited[0]
                state.opt_insert(node)
            return state
        
        initial_solution = construct_initial_solution(use_greedy=True)
        evaluation = evaluate_operator(simple_repair, initial_solution, "test")
        
        self.assertIn('objective', evaluation)
        self.assertIn('gap', evaluation)
        self.assertIn('runtime', evaluation)
        self.assertIn('feasible', evaluation)
        self.assertIsInstance(evaluation['objective'], float)
        self.assertIsInstance(evaluation['gap'], float)
        self.assertIsInstance(evaluation['runtime'], float)
        self.assertIsInstance(evaluation['feasible'], bool)

class TestPCTSPDataLoading(unittest.TestCase):
    """Test PCTSP data loading"""
    
    def test_load_instances(self):
        """Test loading instances"""
        # This test will only pass if the data files exist
        try:
            instances = load_instances(problem_size=20)
            if instances:  # Only test if instances are available
                self.assertIsInstance(instances, list)
                self.assertGreater(len(instances), 0)
                
                # Test first instance
                instance = instances[0]
                self.assertIsInstance(instance, PCTSPData)
                self.assertEqual(instance.size, 20)
                self.assertIsInstance(instance.depot, np.ndarray)
                self.assertIsInstance(instance.locations, np.ndarray)
                self.assertIsInstance(instance.penalties, np.ndarray)
                self.assertIsInstance(instance.prizes, np.ndarray)
        except Exception:
            # Skip test if data files are not available
            self.skipTest("PCTSP data files not available")

def run_tests():
    """Run all tests"""
    print("🧪 Running EoH-PCTSP Test Suite")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPCTSPBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestPCTSPPrompts))
    suite.addTests(loader.loadTestsFromTestCase(TestPCTSPOperators))
    suite.addTests(loader.loadTestsFromTestCase(TestPCTSPDataLoading))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    
    return success

if __name__ == "__main__":
    run_tests() 