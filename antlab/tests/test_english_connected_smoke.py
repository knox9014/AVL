import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from antlab import english_connected_smoke as smoke


class EnglishSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name) / 'smoke'
        cls.report = smoke.run(cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_balanced_heldout_inputs_and_teacher_eos(self):
        data = smoke.make_data()
        train_ids = set()
        for split in ('train', 'heldout'):
            rows = data[split]
            self.assertEqual(sum(row['answer'] == 'YES' for row in rows), len(rows) // 2)
            for row in rows:
                expected = hashlib.sha256(json.dumps(row['local_texts'], separators=(',', ':')).encode()).hexdigest()
                self.assertEqual(row['id'], expected)
                self.assertEqual(len(row['local_texts']), 2)
                self.assertEqual(row['answer'], 'YES' if row['fact_state'] == 'on' else 'NO')
            ids = {row['id'] for row in rows}
            self.assertEqual(len(ids), len(rows))
            if split == 'train':
                train_ids = ids
            else:
                self.assertFalse(train_ids & ids)
        inputs, targets = smoke.teacher_batch([{'answer': 'YES'}, {'answer': 'NO'}])
        self.assertEqual(inputs.tolist(), [[1, 89, 69, 83], [1, 78, 79, 2]])
        self.assertEqual(targets.tolist(), [[89, 69, 83, 2], [78, 79, 2, 0]])

    def test_checkpoint_roundtrip_generates_without_gold(self):
        verification = smoke.verify(self.folder)
        self.assertEqual(verification['status'], 'complete')
        self.assertTrue(self.report['smoke_only'])
        self.assertEqual(self.report['training']['steps'], 400)
        checkpoint = torch.load(self.folder / 'checkpoint.pt', weights_only=True)
        self.assertEqual(checkpoint['parameters'], 98928)
        original = smoke.load_model(self.folder)
        data = smoke.make_data()['heldout']
        changed_gold = [{**row, 'answer': 'NO' if row['answer'] == 'YES' else 'YES'} for row in data]
        first = smoke.evaluate(original, data, 'connected')
        second = smoke.evaluate(original, changed_gold, 'connected')
        self.assertEqual(first['token_ids'], second['token_ids'])
        self.assertEqual(first['texts'], second['texts'])
        wellformed = sum(ids in ([89, 69, 83, 2], [78, 79, 2]) for ids in first['token_ids'])
        self.assertEqual(first['correct'] + second['correct'], wellformed)
        blocked = self.report['evaluations']['heldout']['blocked']
        swapped = self.report['evaluations']['heldout']['blocked_swapped']
        self.assertEqual(blocked['token_ids'], swapped['token_ids'])

    def test_no_overwrite(self):
        with self.assertRaises(FileExistsError):
            smoke.run(self.folder)

    def test_forged_generated_outputs_rejected_even_with_updated_hash(self):
        path, manifest_path = self.folder / 'report.json', self.folder / 'manifest.json'
        old_report, old_manifest = path.read_bytes(), manifest_path.read_bytes()
        try:
            report = json.loads(old_report)
            report['evaluations']['heldout']['connected']['token_ids'][0] = [89, 69, 83, 2]
            report['evaluations']['heldout']['connected']['texts'][0] = 'forged'
            smoke.write_json(path, report)
            manifest = json.loads(old_manifest)
            manifest['files']['report.json'] = smoke.digest(path)
            smoke.write_json(manifest_path, manifest)
            with self.assertRaises(ValueError):
                smoke.verify(self.folder)
        finally:
            path.write_bytes(old_report)
            manifest_path.write_bytes(old_manifest)

    def test_checkpoint_tamper_rejected(self):
        path = self.folder / 'checkpoint.pt'
        old = path.read_bytes()
        try:
            path.write_bytes(old + b'tampered')
            with self.assertRaises(ValueError):
                smoke.verify(self.folder)
        finally:
            path.write_bytes(old)

    def test_forged_parameter_update_record_rejected(self):
        path, manifest_path = self.folder / 'report.json', self.folder / 'manifest.json'
        old_report, old_manifest = path.read_bytes(), manifest_path.read_bytes()
        try:
            report = json.loads(old_report)
            report['training']['changed_parameter_tensors'].pop()
            smoke.write_json(path, report)
            manifest = json.loads(old_manifest)
            manifest['files']['report.json'] = smoke.digest(path)
            smoke.write_json(manifest_path, manifest)
            with self.assertRaises(ValueError):
                smoke.verify(self.folder)
        finally:
            path.write_bytes(old_report)
            manifest_path.write_bytes(old_manifest)


if __name__ == '__main__':
    unittest.main()
