import tempfile
import unittest
from pathlib import Path
import torch
from antlab.semantic_v2_model import SemanticNet
from antlab.semantic_v2_demo import transmit_segments, receive_packet, load_checkpoint


class SemanticV2DemoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(44)
        self.model = SemanticNet().eval()

    def test_packet_boundary_and_unsupported_original_retention(self):
        texts = ['It is certain that the lamp is on.', 'The budget is 17 USD.',
                 'Please ensure that the door is not closed.']
        sent = transmit_segments(self.model, texts)
        self.assertEqual(sent['sources'], texts)
        self.assertEqual(sent['supported_indices'], [0, 2])
        self.assertEqual(sent['unsupported'], [dict(index=1, status='unsupported', text=texts[1])])
        self.assertEqual(len(sent['packet']), 140)
        frames = receive_packet(self.model, sent['packet'])
        self.assertEqual(len(frames), 2)
        first = transmit_segments(self.model, texts[:1])
        self.assertEqual(frames[0], receive_packet(self.model, first['packet'])[0])

    def test_all_unsupported_no_wire_and_input_errors(self):
        sent = transmit_segments(self.model, ['Unlisted entity is on.'])
        self.assertIsNone(sent['packet'])
        self.assertEqual(sent['unsupported'][0]['status'], 'unsupported')
        for value in ([], 'It is certain that the lamp is on.', [None]):
            with self.assertRaises(ValueError):
                transmit_segments(self.model, value)

    def test_checkpoint_version_seed_and_smoke_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'seed-44.pt'
            checkpoint = dict(format='avl-semantic-study-v2', seed=44,
                              config={'smoke_only': False}, state_dict=self.model.state_dict())
            torch.save(checkpoint, path)
            restored = load_checkpoint(directory, 44)
            sent = transmit_segments(self.model, ['It is certain that the lamp is on.'])
            self.assertEqual(receive_packet(self.model, sent['packet']), receive_packet(restored, sent['packet']))
            for key, value in [('format', 'avl-semantic-study-v1'), ('seed', 55), ('config', {'smoke_only': True})]:
                bad = {**checkpoint, key: value}
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    load_checkpoint(directory, 44)


if __name__ == '__main__':
    unittest.main()
