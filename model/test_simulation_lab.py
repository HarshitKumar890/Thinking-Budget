"""
Comprehensive test suite for the Thinking Budget Simulation Lab.
Validates:
1. All model classes instantiate.
2. All benchmark checkpoints load.
3. CoT inference works.
4. Latent inference works.
5. Hebbian inference works.
6. Task A generation works.
7. Task B generation works.
8. Each supported k works.
9. Invalid k is handled cleanly.
10. Single-example simulation works.
11. Batch evaluation compatibility.
12. Reasoning traces are returned.
13. Parameter counts are correct.
14. CPU execution works.
15. CUDA execution works when available.
16. Latent recurrent state resets correctly.
17. Hebbian state resets correctly.
18. Repeated inference does not contaminate later simulations.
19. Existing benchmark evaluation produces compatible results.
20. Existing test suites continue to pass.
"""

import unittest
from pathlib import Path
import torch

from simulation_lab import (
    SimulationLabEngine,
    ModelRegistry,
    TaskRegistry,
    TaskExample,
    SimulationResult,
    TraceContainer
)
from models_cot_model import AutoregressiveCoT
from models_latent_model import RecurrentLatentReasoner
from models_hebbian_model import HebbianMemory
from data_generator import TaskAPermutationOrbit, TaskBModularRegister

class TestSimulationLab(unittest.TestCase):

    def setUp(self):
        self.engine_cpu = SimulationLabEngine(device='cpu')
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.engine_cuda = SimulationLabEngine(device='cuda')

    def test_01_model_instantiation(self):
        """1. All model classes instantiate."""
        cot = AutoregressiveCoT()
        latent = RecurrentLatentReasoner()
        hebbian = HebbianMemory()
        self.assertIsNotNone(cot)
        self.assertIsNotNone(latent)
        self.assertIsNotNone(hebbian)

    def test_02_all_benchmark_checkpoints_load(self):
        """2. All benchmark checkpoints load."""
        export_base = Path("export/models") if Path("export/models").exists() else Path("model/export/models")
        available_seeds = [s for s in [42, 43, 44, 45, 46] if (export_base / f"cot_task_a_s{s}_aligned/weights.pth").exists()]
        self.assertIn(42, available_seeds, "Authoritative Seed 42 must always be present")
        tasks = ['task_a', 'task_b']
        for s in available_seeds:
            for t in tasks:
                cot_p = export_base / f"cot_{t}_s{s}_aligned/weights.pth"
                latent_p = export_base / f"latent_{t}_s{s}_aligned/weights.pth"
                self.assertTrue(cot_p.exists(), f"Missing checkpoint {cot_p}")
                self.assertTrue(latent_p.exists(), f"Missing checkpoint {latent_p}")

                # Verify loading into fresh models
                cot = AutoregressiveCoT()
                cot.load_state_dict(torch.load(cot_p, map_location='cpu'))
                latent = RecurrentLatentReasoner()
                latent.load_state_dict(torch.load(latent_p, map_location='cpu'))

    def test_03_cot_inference_works(self):
        """3. CoT inference works."""
        res = self.engine_cpu.run_simulation(model='cot', task='permutation', budget_k=4, seed=42)
        self.assertIsInstance(res, SimulationResult)
        self.assertIn(res.predicted_answer, list(range(8)))
        self.assertGreater(len(res.trace['steps']), 0)

    def test_04_latent_inference_works(self):
        """4. Latent inference works."""
        res = self.engine_cpu.run_simulation(model='latent', task='permutation', budget_k=4, seed=42)
        self.assertIsInstance(res, SimulationResult)
        self.assertIn(res.predicted_answer, list(range(8)))
        self.assertEqual(len(res.trace['steps']), 4)

    def test_05_hebbian_inference_works(self):
        """5. Hebbian inference works."""
        res = self.engine_cpu.run_simulation(model='hebbian', task='associative', budget_k=1)
        self.assertIsInstance(res, SimulationResult)
        self.assertIn('steps', res.trace)
        self.assertEqual(len(res.trace['steps']), 2)

    def test_06_task_a_generation_works(self):
        """6. Task A generation works."""
        ex = self.engine_cpu.generate_example('permutation_orbit', depth=6, seed=123)
        self.assertEqual(ex.depth, 6)
        self.assertEqual(ex.task_name, 'permutation_orbit')
        self.assertEqual(len(ex.ground_truth_trajectory), 6)

    def test_07_task_b_generation_works(self):
        """7. Task B generation works."""
        ex = self.engine_cpu.generate_example('modular_register', depth=4, seed=123)
        self.assertEqual(ex.depth, 4)
        self.assertEqual(ex.task_name, 'modular_register')
        self.assertEqual(len(ex.ground_truth_trajectory), 4)

    def test_08_each_supported_k_works(self):
        """8. Each supported k works."""
        ex = self.engine_cpu.generate_example('permutation_orbit', depth=8, seed=42)
        for k in [1, 2, 4, 8, 12, 16]:
            res_cot = self.engine_cpu.run_simulation(model='cot', task='permutation', budget_k=k, example=ex)
            res_lat = self.engine_cpu.run_simulation(model='latent', task='permutation', budget_k=k, example=ex)
            self.assertEqual(res_cot.budget_k, k)
            self.assertEqual(res_lat.budget_k, k)
            self.assertEqual(len(res_lat.trace['steps']), k)

    def test_09_invalid_k_handled_cleanly(self):
        """9. Invalid k is handled cleanly."""
        with self.assertRaises(ValueError):
            self.engine_cpu.run_simulation(model='cot', task='permutation', budget_k=7)
        with self.assertRaises(ValueError):
            self.engine_cpu.run_simulation(model='hebbian', task='associative', budget_k=4)

    def test_10_single_example_simulation_works(self):
        """10. Single-example simulation works."""
        ex = TaskRegistry.generate_example('permutation_orbit', depth=5, seed=999)
        res = self.engine_cpu.run_simulation(model='cot', task='permutation', budget_k=4, example=ex)
        self.assertEqual(res.input_representation['depth'], 5)
        self.assertEqual(res.expected_answer, ex.expected_answer)

    def test_11_batch_evaluation_compatibility(self):
        """11. Batch evaluation still works."""
        cot = AutoregressiveCoT()
        latent = RecurrentLatentReasoner()
        inp = torch.randint(0, 10, (4, 13))
        res_cot = cot.generate(inp, max_budget_k=4)
        res_lat = latent.infer(inp, budget_k=4)
        self.assertEqual(res_cot['final_ans'].shape[0], 4)
        self.assertEqual(res_lat['final_ans'].shape[0], 4)

    def test_12_reasoning_traces_returned(self):
        """12. Reasoning traces are returned."""
        res_cot = self.engine_cpu.run_simulation(model='cot', task='permutation', budget_k=4, seed=42)
        res_lat = self.engine_cpu.run_simulation(model='latent', task='permutation', budget_k=4, seed=42)
        self.assertIn('steps', res_cot.trace)
        self.assertIn('steps', res_lat.trace)
        self.assertEqual(res_cot.trace['paradigm'], 'autoregressive_cot')
        self.assertEqual(res_lat.trace['paradigm'], 'recurrent_latent')

    def test_13_parameter_counts_are_correct(self):
        """13. Parameter counts are correct."""
        cot = AutoregressiveCoT()
        latent = RecurrentLatentReasoner()
        heb = HebbianMemory()
        self.assertEqual(sum(p.numel() for p in cot.parameters()), 126168)
        self.assertEqual(sum(p.numel() for p in latent.parameters()), 125688)
        self.assertEqual(sum(p.numel() for p in heb.parameters()), 20672)

    def test_14_cpu_execution_works(self):
        """14. CPU execution works."""
        res = self.engine_cpu.run_simulation(model='latent', task='permutation', budget_k=2, seed=42)
        self.assertEqual(res.device, 'cpu')
        self.assertGreater(res.latency_ms, 0.0)

    def test_15_cuda_execution_works_when_available(self):
        """15. CUDA execution works when available."""
        if not self.cuda_available:
            self.skipTest("CUDA not available on this machine.")
        res = self.engine_cuda.run_simulation(model='latent', task='permutation', budget_k=2, seed=42)
        self.assertEqual(res.device, 'cuda')
        self.assertGreater(res.latency_ms, 0.0)

    def test_16_latent_recurrent_state_resets_correctly(self):
        """16. Latent recurrent state resets correctly."""
        ex1 = self.engine_cpu.generate_example('permutation_orbit', depth=4, seed=1)
        ex2 = self.engine_cpu.generate_example('permutation_orbit', depth=4, seed=2)
        res1 = self.engine_cpu.run_simulation(model='latent', task='permutation', budget_k=4, example=ex1)
        res2 = self.engine_cpu.run_simulation(model='latent', task='permutation', budget_k=4, example=ex2)
        # Running res1 again must yield identical output
        res1_repeat = self.engine_cpu.run_simulation(model='latent', task='permutation', budget_k=4, example=ex1)
        self.assertEqual(res1.predicted_answer, res1_repeat.predicted_answer)

    def test_17_hebbian_state_resets_correctly(self):
        """17. Hebbian state resets correctly."""
        adapter = ModelRegistry.create_adapter('hebbian', device=torch.device('cpu'))
        adapter.predict_with_trace([1, 2, 3], budget_k=1)
        # Matrix must be zeroed on reset
        adapter.reset_state()
        self.assertTrue((adapter.model.sigma == 0).all())

    def test_18_repeated_inference_does_not_contaminate(self):
        """18. Repeated inference does not silently contaminate later simulations."""
        ex = self.engine_cpu.generate_example('permutation_orbit', depth=8, seed=42)
        res_a = self.engine_cpu.run_simulation(model='cot', task='permutation', budget_k=4, example=ex)
        # Run arbitrary intervening simulation
        _ = self.engine_cpu.run_simulation(model='latent', task='modular', budget_k=12, seed=42)
        res_b = self.engine_cpu.run_simulation(model='cot', task='permutation', budget_k=4, example=ex)
        self.assertEqual(res_a.predicted_answer, res_b.predicted_answer)
        self.assertEqual(res_a.trace['raw_summary']['tokens_generated'], res_b.trace['raw_summary']['tokens_generated'])

    def test_19_existing_benchmark_compatibility(self):
        """19. Existing benchmark evaluation produces compatible results."""
        pareto = self.engine_cpu.get_pareto_data('permutation')
        self.assertIn('pareto_frontiers', pareto)
        self.assertIn('aggregated_conditions', pareto)
        self.assertGreater(len(pareto['aggregated_conditions']), 0)

    def test_20_simulation_lab_5m_recovery_checkpoints(self):
        """20. 5M adapters load recovery checkpoints and do NOT point to export/models_5m/."""
        cot_5m = ModelRegistry.create_adapter('autoregressive_cot_5m', device=torch.device('cpu'))
        latent_5m = ModelRegistry.create_adapter('recurrent_latent_5m', device=torch.device('cpu'))

        ckpt_cot = cot_5m.load_checkpoint('task_a', seed=42)
        ckpt_latent = latent_5m.load_checkpoint('task_a', seed=42)

        self.assertIn('export/models_5m_recovery', ckpt_cot.replace('\\', '/'))
        self.assertNotIn('export/models_5m/', ckpt_cot.replace('\\', '/') + '/')
        self.assertIn('export/models_5m_recovery', ckpt_latent.replace('\\', '/'))
        self.assertNotIn('export/models_5m/', ckpt_latent.replace('\\', '/') + '/')

        self.assertTrue(Path(ckpt_cot).exists())
        self.assertTrue(Path(ckpt_latent).exists())

    def test_21_simulation_lab_5m_parameter_counts(self):
        """21. 5M adapters have exact parameter counts: 5,000,632 and 5,000,635."""
        cot_5m = ModelRegistry.create_adapter('autoregressive_cot_5m', device=torch.device('cpu'))
        latent_5m = ModelRegistry.create_adapter('recurrent_latent_5m', device=torch.device('cpu'))

        self.assertEqual(cot_5m.count_parameters(), (5000632, 5000632))
        self.assertEqual(latent_5m.count_parameters(), (5000635, 5000635))

    def test_22_simulation_lab_5m_predict_api(self):
        """22. Unified predict and predict_with_trace work on 5M adapters."""
        cot_5m = ModelRegistry.create_adapter('autoregressive_cot_5m', device=torch.device('cpu'))
        latent_5m = ModelRegistry.create_adapter('recurrent_latent_5m', device=torch.device('cpu'))

        dummy_inp = [13, 0, 1, 2, 3, 4, 5, 6, 7, 14, 0, 15, 2]
        pred_cot = cot_5m.predict(dummy_inp, budget_k=2)
        pred_lat = latent_5m.predict(dummy_inp, budget_k=2)

        self.assertIsInstance(pred_cot, int)
        self.assertIsInstance(pred_lat, int)

        trace_cot = cot_5m.predict_with_trace(dummy_inp, budget_k=2, measure_timing=False)
        trace_lat = latent_5m.predict_with_trace(dummy_inp, budget_k=2, measure_timing=False)

        self.assertEqual(trace_cot['predicted_answer'], pred_cot)
        self.assertEqual(trace_lat['predicted_answer'], pred_lat)
        self.assertEqual(trace_cot['trace']['raw_summary']['scale'], '5m_recovery')
        self.assertEqual(trace_lat['trace']['raw_summary']['scale'], '5m_recovery')

    def test_23_simulation_lab_5m_state_isolation(self):
        """23. State reset and repeated inference work correctly on 5M models."""
        ex1 = self.engine_cpu.generate_example('permutation_orbit', depth=4, seed=1)
        res1 = self.engine_cpu.run_simulation(model='cot_5m', task='permutation', budget_k=4, example=ex1)
        res1_repeat = self.engine_cpu.run_simulation(model='cot_5m', task='permutation', budget_k=4, example=ex1)
        self.assertEqual(res1.predicted_answer, res1_repeat.predicted_answer)

    def test_24_simulation_lab_5m_flops(self):
        """24. Analytical FLOPs for 5M match the 5M architecture."""
        from simulation_lab.metrics import compute_analytical_flops
        cot_flops = compute_analytical_flops("autoregressive_cot", 1, d_model=544, dim_feedforward_cot=1168, num_layers=2)
        lat_flops = compute_analytical_flops("recurrent_latent", 1, d_model=544, dim_feedforward_latent=1795)

        self.assertEqual(cot_flops, 9844224.0)
        self.assertEqual(lat_flops, 9824640.0)

    def test_25_simulation_lab_5m_registry_metadata(self):
        """25. Registry metadata indicates Recovery for 5M models."""
        models = {m.model_id: m for m in ModelRegistry.list_models()}
        self.assertIn('autoregressive_cot_5m', models)
        self.assertIn('recurrent_latent_5m', models)

        self.assertIn('Recovery', models['autoregressive_cot_5m'].display_name)
        self.assertIn('Recovery', models['recurrent_latent_5m'].display_name)
        self.assertIn('models_5m_recovery', models['autoregressive_cot_5m'].description)
        self.assertIn('models_5m_recovery', models['recurrent_latent_5m'].description)

    def test_26_simulation_lab_5m_pareto_retrieval(self):
        """26. Engine retrieves 5M recovery Pareto data accurately."""
        data_5m = self.engine_cpu.get_pareto_data('permutation_orbit', scale='5m')
        self.assertIn('pareto_frontiers', data_5m)
        self.assertIn('aggregated_conditions', data_5m)
        self.assertEqual(data_5m['scale'], '5m')
        self.assertGreater(len(data_5m['aggregated_conditions']), 0)

    def test_27_model_registry_caching_and_immutability(self):
        """27. list_models() caches metadata without redundant adapter allocations and prevents cache corruption."""
        from unittest.mock import patch

        instantiated_classes = []
        def make_spy(cls_ref):
            class AdapterSpy(cls_ref):
                def __init__(self, *args, **kwargs):
                    instantiated_classes.append(cls_ref.__name__)
                    super().__init__(*args, **kwargs)
            return AdapterSpy

        # Save original cached state
        saved_cache = ModelRegistry._cached_models
        try:
            # Force cache reset for controlled lifecycle evaluation
            ModelRegistry._cached_models = None

            with patch.dict(ModelRegistry.AVAILABLE_MODELS, {
                mid: {**info, 'adapter_class': make_spy(info['adapter_class'])}
                for mid, info in ModelRegistry.AVAILABLE_MODELS.items()
            }):
                # 1. First uncached call must instantiate each registered model adapter exactly once
                expected_adapter_count = len(ModelRegistry.AVAILABLE_MODELS)
                first_call = ModelRegistry.list_models()
                self.assertEqual(
                    len(instantiated_classes),
                    expected_adapter_count,
                    f"First list_models() call must instantiate all {expected_adapter_count} adapters"
                )
                self.assertEqual(len(first_call), expected_adapter_count)

                # Reset spy tracker to observe subsequent invocations
                instantiated_classes.clear()

                # 2. Repeated call must return identical metadata without any new adapter instantiations
                repeated_call = ModelRegistry.list_models()
                self.assertEqual(
                    len(instantiated_classes),
                    0,
                    "Repeated list_models() must serve from cache and perform 0 adapter instantiations"
                )
                self.assertEqual(first_call, repeated_call)

                # 3. Mutating the returned list does NOT corrupt subsequent calls (shallow copy protection)
                repeated_call.pop()
                subsequent_call = ModelRegistry.list_models()
                self.assertEqual(len(subsequent_call), expected_adapter_count)
                self.assertEqual(subsequent_call, first_call)
        finally:
            # Restore state to prevent contaminating subsequent test cases
            ModelRegistry._cached_models = saved_cache


if __name__ == '__main__':
    unittest.main()
