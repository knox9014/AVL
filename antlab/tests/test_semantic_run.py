import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from antlab import semantic_run as runner


class SemanticRunTests(unittest.TestCase):
    def test_training_and_evaluation_normalize_text_without_changing_raw_data(self):
        rows = runner.make_data()['validation'][:4]
        changed = [{**row, 'text': '\t ' + row['text'].swapcase().replace(' ', '\t  ') + '\t'} for row in rows]
        tokens, lengths = runner._batch(rows)
        normalized_tokens, normalized_lengths = runner._batch(changed)
        self.assertTrue(torch.equal(tokens, normalized_tokens))
        self.assertTrue(torch.equal(lengths, normalized_lengths))
        self.assertTrue(changed[0]['text'].startswith('\t '))
        model = runner.SemanticNet().eval()
        first = runner.evaluate(model, rows, [0] * 9, 11)
        second = runner.evaluate(model, changed, [0] * 9, 11)
        for mode in runner.MODES:
            self.assertEqual(first[mode], second[mode])
        self.assertNotEqual(first['raw_source_bytes'], second['raw_source_bytes'])

    def test_invalid_classes_are_negative_infinity_and_valid_logits_must_be_finite(self):
        logits = torch.zeros(2, 9, 8)
        masked = runner._mask(logits)
        for index, vocab in enumerate(runner.VOCABS):
            self.assertTrue(torch.isfinite(masked[:, index, :len(vocab)]).all())
            self.assertTrue(torch.isneginf(masked[:, index, len(vocab):]).all())
        self.assertEqual(runner._mask(torch.full((2, 9, 8), -1e6)).argmax(-1).tolist(), [[0] * 9] * 2)
        logits[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            runner._mask(logits)

    def test_frozen_full_configuration(self):
        cfg = runner.configuration()
        self.assertEqual(cfg['seeds'], [11, 22, 33])
        self.assertEqual((cfg['steps'], cfg['batch_size'], cfg['max_seconds']), (1500, 128, 1800))
        self.assertFalse(cfg['smoke_only'])
        for key, value in [('steps', 1499), ('max_seconds', 1801), ('learning_rate', .01)]:
            changed = dict(cfg, **{key: value})
            with self.assertRaises(ValueError):
                runner.validate_configuration(changed)

    def test_metrics_confusions_and_absent_classes(self):
        gold = torch.zeros(2, 9, dtype=torch.long)
        pred = gold.clone()
        pred[1, 0] = 1
        result = runner.metrics(gold, pred)
        self.assertEqual(result['exact_frame_accuracy'], .5)
        self.assertEqual(result['fields']['kind']['accuracy'], .5)
        self.assertEqual(result['fields']['kind']['confusion'], [[1, 1], [0, 0]])
        self.assertEqual(result['fields']['kind']['macro_class_accuracy'], .5)
        self.assertEqual(result['fields']['kind']['class_support'], [2, 0])

    def test_serialized_receiver_and_controls(self):
        rows = runner.make_data()['validation'][:4]
        model = runner.SemanticNet().eval()
        majority = [0] * 9
        original_unpack = runner.unpack_vectors
        with patch.object(runner, 'unpack_vectors', wraps=original_unpack) as unpack:
            result = runner.evaluate(model, rows, majority, 11, batch_size=2)
        self.assertEqual(unpack.call_count, 2)
        self.assertEqual(result['wire_bytes'], 2 * 12 + 4 * 64)
        self.assertEqual(result['gold_oracle']['exact_frame_accuracy'], 1.)
        self.assertEqual(result['query_only']['predictions'], [[0] * 9] * 4)
        changed = [{**row, 'labels': [(label + 1) % len(vocab) for label, vocab in zip(row['labels'], runner.VOCABS)]} for row in rows]
        other = runner.evaluate(model, changed, majority, 11, batch_size=2)
        for mode in ('transmitted', 'zero', 'shuffled', 'query_only'):
            self.assertEqual(result[mode]['predictions'], other[mode]['predictions'])
        # Altering only the unpacked payload must alter the receiver input. A
        # hidden encoder-state or pre-codec shortcut would fail this equality.
        with patch.object(runner, 'unpack_vectors', side_effect=lambda packet: torch.zeros_like(original_unpack(packet))):
            zero_transport = runner.evaluate(model, rows, majority, 11, batch_size=2)
        self.assertEqual(zero_transport['transmitted'], zero_transport['zero'])

    def test_total_deadline_checked_before_encoder_and_training(self):
        rows = runner.make_data()['train'][:4]
        model = runner.SemanticNet()
        with patch.object(runner.time, 'monotonic', return_value=10):
            with self.assertRaises(TimeoutError):
                runner.evaluate(model, rows, [0] * 9, 11, deadline=9)
            with self.assertRaises(TimeoutError):
                runner.train(model, rows, runner.configuration(smoke=True), 11, deadline=9)

    def test_smoke_reload_replay_and_tampering(self):
        cfg = runner.configuration(smoke=True)
        cfg.update(seeds=[11], steps=2, batch_size=4)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'run'
            with patch.object(runner, 'configuration', return_value=cfg):
                report = runner.run(folder, smoke=True)
            self.assertTrue(report['smoke_only'])
            self.assertFalse(report['study_passed'])
            self.assertEqual(report['seeds']['11']['training']['steps'], 2)
            verified = runner.verify(folder)
            self.assertEqual(verified['status'], 'verified')
            self.assertEqual(verified['seeds_verified'], [11])
            with self.assertRaises(FileExistsError):
                runner.run(folder, smoke=True)
            checkpoint = folder / 'seed-11.pt'
            original = checkpoint.read_bytes()
            checkpoint.write_bytes(original + b'tampered')
            with self.assertRaises(ValueError):
                runner.verify(folder)
            checkpoint.write_bytes(original)
            config_path = folder / 'config.json'
            saved = config_path.read_bytes()
            altered = copy.deepcopy(cfg)
            altered['max_seconds'] = 1801
            config_path.write_text(json.dumps(altered), encoding='utf-8')
            with self.assertRaises(ValueError):
                runner.verify(folder)
            config_path.write_bytes(saved)
            report_path = folder / 'report.json'
            altered_report = json.loads(report_path.read_text(encoding='utf-8'))
            altered_report['seeds']['11']['training']['seconds'] = 1801
            report_path.write_text(json.dumps(altered_report), encoding='utf-8')
            with self.assertRaises(ValueError):
                runner.verify(folder)

    def test_rehashed_training_metadata_tampering_is_rejected(self):
        cfg = runner.configuration(smoke=True)
        cfg.update(seeds=[11], steps=2, batch_size=4)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'run'
            with patch.object(runner, 'configuration', return_value=cfg):
                runner.run(folder, smoke=True)
            original_report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
            original_manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
            variants = []
            wrong_delta = copy.deepcopy(original_report)
            wrong_delta['seeds']['11']['training']['parameter_delta_l2'] *= 2
            variants.append(wrong_delta)
            wrong_steps = copy.deepcopy(original_report)
            wrong_steps['seeds']['11']['training']['logs'][1]['step'] = 1
            variants.append(wrong_steps)
            wrong_loss = copy.deepcopy(original_report)
            wrong_loss['seeds']['11']['training']['logs'][0]['loss'] = -1.
            variants.append(wrong_loss)
            wrong_gradient = copy.deepcopy(original_report)
            wrong_gradient['seeds']['11']['training']['logs'][0]['gradient_norm'] = float('inf')
            variants.append(wrong_gradient)
            for changed in variants:
                (folder / 'report.json').write_text(json.dumps(changed), encoding='utf-8')
                manifest = copy.deepcopy(original_manifest)
                manifest['files']['report.json'] = runner.digest(folder / 'report.json')
                (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaises(ValueError):
                    runner.verify(folder)


if __name__ == '__main__':
    unittest.main()
