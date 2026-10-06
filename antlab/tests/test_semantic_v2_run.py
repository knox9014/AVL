import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from antlab import semantic_v2_run as runner


class SemanticV2RunTests(unittest.TestCase):
    def test_frozen_preset_and_smoke(self):
        cfg = runner.configuration()
        self.assertEqual(cfg['seeds'], [44, 55, 66])
        self.assertEqual((cfg['steps'], cfg['batch_size'], cfg['max_seconds']), (1800, 128, 1800))
        self.assertEqual(cfg['consistency_weight'], .05)
        self.assertLessEqual(cfg['parameters'], 110000)
        for key, value in [('steps', 1801), ('learning_rate', .01), ('consistency_weight', 0.)]:
            with self.assertRaises(ValueError):
                runner.validate_configuration(dict(cfg, **{key: value}))

    def test_pair_sampling_stays_in_train_frame_and_changes_surface(self):
        rows = runner.make_data()['train']
        candidates = runner.pair_candidates(rows)
        anchors, partners = runner.sample_pairs(candidates, 128, torch.Generator().manual_seed(44))
        for first, second in zip(anchors.tolist(), partners.tolist()):
            self.assertEqual(rows[first]['labels'], rows[second]['labels'])
            self.assertNotEqual(runner.normalize_text(rows[first]['text']), runner.normalize_text(rows[second]['text']))
        with self.assertRaises(ValueError):
            runner.pair_candidates(rows[:1])

    def test_joint_is_a_required_gate(self):
        fields = {field: {'accuracy': .96} for field in ('polarity', 'certainty', 'condition')}
        evaluations = {split: {'transmitted': {'exact_frame_accuracy': .96, 'fields': copy.deepcopy(fields)}}
                       for split in runner.SPLITS}
        evaluations['joint']['transmitted']['exact_frame_accuracy'] = .94
        gates = runner._gates(evaluations)
        self.assertFalse(gates['joint'])
        self.assertTrue(all(gates[split] for split in ('validation', 'combination', 'phrasing')))

    def test_global_shuffle_byte_transport_and_direct_equivalence(self):
        rows = runner.make_data()['validation'][:5]
        model = runner.SemanticNet().eval()
        decoded = []
        original = model.decode
        def record(vectors):
            decoded.append(vectors.clone())
            return original(vectors)
        with patch.object(model, 'decode', side_effect=record):
            result = runner.evaluate(model, rows, [0] * 9, 44, batch_size=2)
        self.assertTrue(result['codec_vectors_equal'])
        self.assertEqual(result['transmitted'], result['direct_roundtrip'])
        indices = result['shuffled_indices']
        self.assertEqual(sorted(indices), list(range(5)))
        self.assertTrue(all(index != source for index, source in enumerate(indices)))
        # Four decodes per batch: transmitted, zero, globally shuffled, direct.
        received = torch.cat(decoded[0::4])
        shuffled = torch.cat(decoded[2::4])
        self.assertTrue(torch.equal(shuffled, received[indices]))
        self.assertEqual(result['wire_bytes'], 3 * 12 + 5 * 64)
        original_unpack = runner.unpack_vectors
        with patch.object(runner, 'unpack_vectors', side_effect=lambda packet: torch.zeros_like(original_unpack(packet))):
            changed = runner.evaluate(model, rows, [0] * 9, 44, batch_size=2)
        self.assertFalse(changed['codec_vectors_equal'])
        self.assertEqual(changed['transmitted'], changed['zero'])

    def test_smoke_replay_rejects_rehashed_metrics_and_training_metadata(self):
        cfg = runner.configuration(smoke=True)
        cfg.update(seeds=[44], steps=2, batch_size=4)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'v2'
            with patch.object(runner, 'configuration', return_value=cfg):
                report = runner.run(folder, smoke=True)
            self.assertTrue(report['smoke_only'])
            self.assertFalse(report['study_passed'])
            self.assertEqual(set(report['seeds']['44']['evaluations']), set(runner.SPLITS))
            self.assertEqual(set(report['seeds']['44']['gates']), {'validation', 'combination', 'phrasing', 'joint'})
            self.assertEqual(runner.verify(folder)['seeds_verified'], [44])
            original = copy.deepcopy(report)
            manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
            variants = []
            altered = copy.deepcopy(original)
            altered['seeds']['44']['training']['logs'][0]['consistency_loss'] += 1.
            variants.append(altered)
            altered = copy.deepcopy(original)
            altered['seeds']['44']['training']['parameter_delta_l2'] *= 2
            variants.append(altered)
            altered = copy.deepcopy(original)
            altered['seeds']['44']['evaluations']['validation']['shuffled_indices'][0] = 0
            variants.append(altered)
            altered = copy.deepcopy(original)
            altered['seeds']['44']['evaluations']['joint']['query_only']['exact_frame_accuracy'] = 1.
            variants.append(altered)
            for altered in variants:
                runner.write_json(folder / 'report.json', altered)
                current = copy.deepcopy(manifest)
                current['files']['report.json'] = runner.digest(folder / 'report.json')
                runner.write_json(folder / 'manifest.json', current)
                with self.assertRaises(ValueError):
                    runner.verify(folder)
            # Even a refreshed manifest cannot turn an altered saved budget
            # into this frozen study, or supply a differently configured model.
            runner.write_json(folder / 'report.json', original)
            current = copy.deepcopy(manifest)
            altered_cfg = dict(cfg, max_seconds=1801)
            runner.write_json(folder / 'config.json', altered_cfg)
            current['files']['config.json'] = runner.digest(folder / 'config.json')
            runner.write_json(folder / 'manifest.json', current)
            with self.assertRaises(ValueError):
                runner.verify(folder)
            runner.write_json(folder / 'config.json', cfg)
            checkpoint_path = folder / 'seed-44.pt'
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            checkpoint['steps_completed'] = 3
            torch.save(checkpoint, checkpoint_path)
            current = copy.deepcopy(manifest)
            current['files']['seed-44.pt'] = runner.digest(checkpoint_path)
            runner.write_json(folder / 'manifest.json', current)
            with self.assertRaises(ValueError):
                runner.verify(folder)

    def test_deadline_and_normalization(self):
        rows = runner.make_data()['validation'][:3]
        changed = [{**row, 'text': '\t' + row['text'].swapcase().replace(' ', '\t ')} for row in rows]
        first, first_lengths = runner._batch(rows)
        second, second_lengths = runner._batch(changed)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first_lengths, second_lengths))
        with patch.object(runner.time, 'monotonic', return_value=10):
            with self.assertRaises(TimeoutError):
                runner.evaluate(runner.SemanticNet(), rows, [0] * 9, 44, deadline=9)


if __name__ == '__main__':
    unittest.main()
