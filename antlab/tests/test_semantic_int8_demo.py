import unittest
from pathlib import Path
from antlab.semantic_v2_demo import load_checkpoint
from antlab import semantic_int8_demo


class Int8DemoTests(unittest.TestCase):
    def test_actual_checkpoint_receives_lossy_bytes_and_preserves_unsupported_source(self):
        model = load_checkpoint(Path(__file__).resolve().parents[1]/'runs/semantic-v2-20261006')
        result = semantic_int8_demo.transmit(model, ['It is certain that the lamp is on.', 'unsupported prose'])
        self.assertEqual(result['wire_bytes'], 32)
        self.assertEqual(result['records'][0]['meaning'], dict(kind='fact', subject='lamp', predicate='on',
                         polarity='positive', certainty='certain', condition='none',
                         amount='none', comparator='none', unit='none'))
        self.assertEqual(result['records'][1]['status'], 'unsupported')
        self.assertEqual(result['records'][1]['source'], 'unsupported prose')


if __name__ == '__main__':
    unittest.main()
