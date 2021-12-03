"""
Integration tests for SSA module.
"""

import unittest

import numpy as np
import pytest
from pyclits.ssa import SSA

DEFAULT_SEED = 42


class TestSSA(unittest.TestCase):
    def generate_ts(self, ndim=1):
        np.random.seed(DEFAULT_SEED)
        return np.random.rand(100, ndim).squeeze()

    def test_init(self):
        ssa = SSA(self.generate_ts(ndim=2), M=4)
        self.assertTrue(isinstance(ssa, SSA))
        self.assertTrue(ssa.compute_rc)
        self.assertFalse(ssa.rotated)
        self.assertEqual(ssa.M, 4)
        self.assertEqual(ssa.n, 100)
        self.assertEqual(ssa.d, 2)
        np.testing.assert_equal(ssa.X, self.generate_ts(ndim=2))

        ssa = SSA(self.generate_ts(ndim=1), M=4)
        self.assertEqual(ssa.n, 100)
        self.assertEqual(ssa.d, 1)

    def test_shift(self):
        SHIFT = 4

        ts = self.generate_ts(ndim=1)
        ssa = SSA(ts, M=4)
        ts_shift = ssa._shift(ts, n=SHIFT, order="forward")
        self.assertEqual(ts.shape[0], len(ts_shift))
        np.testing.assert_equal(ts[SHIFT:], ts_shift[:-SHIFT])

        ts_shift = ssa._shift(ts, n=SHIFT, order="reversed")
        self.assertEqual(ts.shape[0], len(ts_shift))
        np.testing.assert_equal(ts[:-SHIFT], ts_shift[SHIFT:])

        with pytest.raises(ValueError):
            ts_shift = ssa._shift(ts, n=SHIFT, order="161")

    def test_get_cov_matrix(self):
        ts = self.generate_ts(ndim=2)
        ssa = SSA(ts, M=4)
        C, aug_x = ssa._get_cov_matrix(ts, rank_def=True)
        self.assertTrue(isinstance(C, np.ndarray))
        self.assertTrue(isinstance(aug_x, np.ndarray))
        self.assertTupleEqual(C.shape, (ssa.n, ssa.n))
        self.assertTupleEqual(aug_x.shape, (ssa.n, ssa.d * ssa.M))

        C, aug_x = ssa._get_cov_matrix(ts, rank_def=False)
        self.assertTrue(isinstance(C, np.ndarray))
        self.assertTrue(isinstance(aug_x, np.ndarray))
        self.assertTupleEqual(C.shape, (ssa.d * ssa.M, ssa.d * ssa.M))
        self.assertTupleEqual(aug_x.shape, (ssa.n, ssa.d * ssa.M))

    def test_run_ssa(self):
        ts = self.generate_ts(ndim=2)
        ssa = SSA(ts, M=4, compute_rc=False)
        lam, eigenvec, pc = ssa.run()
        self.assertTrue(isinstance(lam, np.ndarray))
        self.assertTrue(isinstance(eigenvec, np.ndarray))
        self.assertTrue(isinstance(pc, np.ndarray))
        self.assertTupleEqual(lam.shape, (ssa.d * ssa.M,))
        self.assertTupleEqual(eigenvec.shape, (ssa.d * ssa.M, ssa.d * ssa.M))
        self.assertTupleEqual(pc.shape, (ssa.n, ssa.d * ssa.M))

        ssa = SSA(ts, M=4, compute_rc=True)
        lam2, eigenvec2, pc2, rc = ssa.run()
        self.assertTrue(isinstance(lam2, np.ndarray))
        self.assertTrue(isinstance(eigenvec2, np.ndarray))
        self.assertTrue(isinstance(pc2, np.ndarray))
        self.assertTrue(isinstance(rc, np.ndarray))
        self.assertTupleEqual(lam2.shape, (ssa.d * ssa.M,))
        self.assertTupleEqual(eigenvec2.shape, (ssa.d * ssa.M, ssa.d * ssa.M))
        self.assertTupleEqual(pc2.shape, (ssa.n, ssa.d * ssa.M))
        self.assertTupleEqual(rc.shape, (ssa.n, ssa.d, ssa.d * ssa.M))

        np.testing.assert_allclose(lam, lam2)
        np.testing.assert_allclose(eigenvec, eigenvec2)
        np.testing.assert_allclose(pc, pc2)

    def test_get_structured_varimax_rotation_matrix(self):
        NUM_EIG = 3

        ts = self.generate_ts(ndim=2)
        ssa = SSA(ts, M=4, compute_rc=True)
        _ = ssa.run()
        ssa.num_eigen = NUM_EIG
        rot_mat = ssa._get_structured_varimax_rotation_matrix()
        self.assertTrue(isinstance(rot_mat, np.ndarray))
        self.assertTupleEqual(rot_mat.shape, (NUM_EIG, NUM_EIG))

    def test_get_orthomax_rotation_matrix(self):
        NUM_EIG = 3

        ts = self.generate_ts(ndim=2)
        ssa = SSA(ts, M=4, compute_rc=True)
        _ = ssa.run()
        ssa.num_eigen = NUM_EIG
        rot_mat = ssa._get_orthomax_rotation_matrix()
        self.assertTrue(isinstance(rot_mat, np.ndarray))
        self.assertTupleEqual(rot_mat.shape, (NUM_EIG, NUM_EIG))

    def test_apply_varimax(self):
        NUM_EIG = 3

        ts = self.generate_ts(ndim=2)
        ssa = SSA(ts, M=4, compute_rc=False)
        _ = ssa.run()
        lam, eigenvec, pc = ssa.apply_varimax(
            NUM_EIG, structured=True, sort_lam=True
        )
        self.assertTrue(ssa.rotated)
        self.assertTrue(isinstance(lam, np.ndarray))
        self.assertTrue(isinstance(eigenvec, np.ndarray))
        self.assertTrue(isinstance(pc, np.ndarray))
        self.assertTupleEqual(lam.shape, (NUM_EIG,))
        self.assertTupleEqual(eigenvec.shape, (ssa.d * ssa.M, NUM_EIG))
        self.assertTupleEqual(pc.shape, (ssa.n, NUM_EIG))

        lam_, eigenvec_, pc_ = ssa.apply_varimax(
            NUM_EIG, structured=False, sort_lam=False
        )
        self.assertTrue(ssa.rotated)
        self.assertTrue(isinstance(lam_, np.ndarray))
        self.assertTrue(isinstance(eigenvec_, np.ndarray))
        self.assertTrue(isinstance(pc_, np.ndarray))
        self.assertTupleEqual(lam_.shape, (NUM_EIG,))
        self.assertTupleEqual(eigenvec_.shape, (ssa.d * ssa.M, NUM_EIG))
        self.assertTupleEqual(pc_.shape, (ssa.n, NUM_EIG))

        ssa = SSA(ts, M=4, compute_rc=True)
        _ = ssa.run()
        lam2, eigenvec2, pc2, rc = ssa.apply_varimax(
            NUM_EIG, structured=True, sort_lam=True
        )
        self.assertTrue(ssa.rotated)
        self.assertTrue(isinstance(lam2, np.ndarray))
        self.assertTrue(isinstance(eigenvec2, np.ndarray))
        self.assertTrue(isinstance(pc2, np.ndarray))
        self.assertTrue(isinstance(rc, np.ndarray))
        self.assertTupleEqual(lam2.shape, (NUM_EIG,))
        self.assertTupleEqual(eigenvec2.shape, (ssa.d * ssa.M, NUM_EIG))
        self.assertTupleEqual(pc2.shape, (ssa.n, NUM_EIG))
        self.assertTupleEqual(rc.shape, (ssa.n, ssa.d, ssa.d * ssa.M))

        np.testing.assert_allclose(lam, lam2)
        np.testing.assert_allclose(eigenvec, eigenvec2)
        np.testing.assert_allclose(pc, pc2)


if __name__ == "__main__":
    unittest.main()
