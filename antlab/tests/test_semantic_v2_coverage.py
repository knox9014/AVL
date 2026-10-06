import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from antlab.semantic_v2_data import is_supported, make_data
from antlab.semantic_v2_run import configuration
from antlab.semantic_v2_coverage import make_audit_data, run


class SemanticV2CoverageTests(unittest.TestCase):
    def test_full_grammar_and_unmeasured_membership(self):
        audit = make_audit_data()
        primary = {r['id'] for rows in make_data().values() for r in rows}
        all_ids = {r['id'] for r in audit['all']}
        unseen = {r['id'] for r in audit['unmeasured']}
        self.assertEqual(len(primary), 4568)
        self.assertEqual(len(audit['all']), 5376)
        self.assertEqual(len(all_ids), 5376)
        self.assertEqual(len(audit['unmeasured']), 808)
        self.assertEqual(unseen, all_ids - primary)
        self.assertFalse(unseen & primary)
        self.assertEqual(all_ids, primary | unseen)
        self.assertTrue(all(is_supported(r['text']) for r in audit['all']))
        self.assertEqual(audit, make_audit_data())

    def test_report_is_posthoc_new_file_and_main_study_unchanged(self):
        cfg = configuration()
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary) / 'study'
            study.mkdir()
            report = dict(format=cfg['format'], status='complete', smoke_only=False,
                          study_passed=False, seeds={str(s): {} for s in cfg['seeds']},
                          source_hashes={})
            (study / 'config.json').write_text(json.dumps(cfg))
            (study / 'report.json').write_text(json.dumps(report))
            for seed in cfg['seeds']:
                (study / f'seed-{seed}.pt').write_bytes(b'fake checkpoint for mocked load')
            before = (study / 'report.json').read_bytes()
            output = Path(temporary) / 'audit.json'
            def evaluation(model, rows, majority, seed, batch_size):
                return {'transmitted': {'examples': len(rows), 'predictions': [[0]*9],
                                        'exact_frame_accuracy': .5}, 'wire_bytes': len(rows)*64+12}
            with patch('antlab.semantic_v2_coverage.verify', return_value={'status': 'verified'}) as verifier, \
                 patch('antlab.semantic_v2_coverage.load_checkpoint', return_value=object()) as loader, \
                 patch('antlab.semantic_v2_coverage.evaluate', side_effect=evaluation):
                result = run(study, output)
                verifier.assert_called_once_with(study)
                self.assertEqual(loader.call_count, 3)
                self.assertEqual(result['status'], 'posthoc_coverage')
                self.assertFalse(result['preregistered'])
                self.assertFalse(result['changes_primary_gates'])
                self.assertFalse(result['primary_study_passed'])
                self.assertEqual(result['counts'], {'all': 5376, 'primary': 4568, 'unmeasured': 808})
                self.assertEqual(json.loads(output.read_text()), result)
                self.assertNotIn('predictions', result['seeds']['44']['evaluations']['all']['transmitted'])
                self.assertEqual((study / 'report.json').read_bytes(), before)
                with self.assertRaises(FileExistsError): run(study, output)

    def test_incomplete_study_is_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            (study / 'config.json').write_text(json.dumps(configuration()))
            (study / 'report.json').write_text(json.dumps(dict(status='running')))
            output = study / 'audit.json'
            with patch('antlab.semantic_v2_coverage.verify') as verifier:
                with self.assertRaises(ValueError): run(study, output)
                verifier.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
