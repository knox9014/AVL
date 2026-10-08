import unittest
import torch
from antlab.semantic_binding_data import make_data, canonicalize
from antlab.semantic_alignment_pilot import fit_ridge, apply_ridge, select_calibration

class AlignmentTests(unittest.TestCase):
    def test_affine_fit_and_real_wire_output(self):
        x=torch.eye(16,dtype=torch.float32)
        x=torch.cat((x,-x))
        y=2*x+1
        weights=fit_ridge(x,y)
        predicted=apply_ridge(x,weights)
        self.assertEqual(weights.shape,(17,16))
        self.assertEqual(predicted.dtype,torch.float32)
        self.assertLess(float((predicted-y).abs().max()),.02)
        with self.assertRaises(ValueError):
            fit_ridge(x,torch.full_like(y,float('nan')))
        with self.assertRaises(ValueError):
            fit_ridge(x,y[:2])

    def test_calibration_balanced_nested_and_unique(self):
        rows=make_data()['train']
        subsets=[select_calibration(rows,n,101) for n in (4,8,16)]
        for n,selected in zip((4,8,16),subsets):
            self.assertEqual(len(selected),n)
            self.assertEqual([r['target'] for r in selected],list(range(4))*(n//4))
            self.assertEqual(len({tuple(canonicalize(r['text'],r['table'])) for r in selected}),n)
        self.assertEqual(subsets[0],subsets[1][:4])
        self.assertEqual(subsets[1],subsets[2][:8])
        with self.assertRaises(ValueError):
            select_calibration(rows,5,101)
        with self.assertRaises(ValueError):
            select_calibration(rows[:1],4,101)

if __name__=='__main__':
    unittest.main()
