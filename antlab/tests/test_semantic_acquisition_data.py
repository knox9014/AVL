import unittest
from antlab.semantic_binding_data import make_data,canonicalize
from antlab.semantic_acquisition_data import training_bank,select_support,summarize_thresholds


class AcquisitionDataTests(unittest.TestCase):
    def test_train_only_distinct_balanced_nested(self):
        data=make_data(); bank=training_bank(data['train'])
        self.assertEqual(len(bank),16)
        for seed in (101,202,303):
            previous=set()
            for budget in (4,8,16):
                support=select_support(bank,budget,seed)
                forms={tuple(canonicalize(r['text'],r['table'])) for r in support}
                self.assertEqual(len(forms),budget)
                self.assertTrue(previous<=forms)
                self.assertEqual([sum(r['target']==c for r in support) for c in range(4)],[budget//4]*4)
                self.assertTrue(all(r in data['train'] for r in support))
                previous=forms
                self.assertEqual(support,select_support(bank,budget,seed))

    def test_reject_conflicting_form_and_bad_budget(self):
        rows=make_data()['train']; altered=dict(rows[0],target=rows[0]['target']^1)
        with self.assertRaises(ValueError): training_bank(rows+[altered])
        with self.assertRaises(ValueError): select_support(training_bank(rows),5,101)

    def test_censoring_preserves_failures(self):
        summary=summarize_thresholds([1,None,5,None])
        self.assertEqual(summary,dict(trials=4,reached=2,fraction=.5,conditional_median_step=3.))
        self.assertIsNone(summarize_thresholds([None])['conditional_median_step'])


if __name__=='__main__': unittest.main()

