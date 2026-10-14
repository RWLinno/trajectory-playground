import unittest
import numpy as np
from src.experiment import valid_paths, paths, metrics, conditions, reference_amplitudes

class GeometryTests(unittest.TestCase):
    def test_segment_interior_collision(self):
        valid, clearance = valid_paths(np.array([[[-.9,0],[.9,0]]]), [.2,0,0])
        self.assertFalse(valid[0]); self.assertAlmostEqual(clearance[0], -.2)
    def test_two_valid_modes_and_endpoints(self):
        c = [.25,0,.2]
        p = paths([.45,-.45], c)
        self.assertTrue(valid_paths(p,c)[0].all())
        self.assertEqual(metrics(p,c)['valid_modes'], 2)
        np.testing.assert_allclose(p[:,0], [[-.9,-.18]]*2)
        np.testing.assert_allclose(p[:,-1], [[.9,.18]]*2)
    def test_noise_is_not_valid_diversity(self):
        p = paths([.01,-.01], [.25,0,0])
        self.assertEqual(metrics(p,[.25,0,0])['valid_modes'],0)
    def test_reference_distribution_is_feasible(self):
        rng=np.random.default_rng(7); cs=conditions(rng,1000)
        amplitudes=reference_amplitudes(rng,cs)
        self.assertTrue(all(valid_paths(paths([a],c),c)[0][0] for a,c in zip(amplitudes,cs)))

if __name__ == '__main__': unittest.main()
