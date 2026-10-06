import tempfile
import unittest
import subprocess
import contextlib
import io
import json
from unittest.mock import patch
from pathlib import Path
from antlab.publication_guard import scan_bytes, inspect_git, main


class PublicationGuardTests(unittest.TestCase):
    def test_private_values_report_only_location_and_category(self):
        secret = 'sk-' + 'X' * 40
        text = '\n'.join(['C:' + '\\Users\\' + 'person\\file.txt',
                          '/' + 'home/' + 'person/file.txt',
                          'person' + '@mail.invalid', secret,
                          '-----BEGIN ' + 'PRIVATE KEY-----'])
        findings, skipped = scan_bytes('sample.txt', text.encode(), identities=())
        self.assertIsNone(skipped)
        self.assertEqual({f['category'] for f in findings},
                         {'local_user_path', 'email_address', 'api_token', 'private_key'})
        self.assertEqual(set(findings[0]), {'file', 'line', 'category'})
        self.assertNotIn(secret, str(findings))
        self.assertNotIn('person', str(findings))

    def test_reserved_email_is_allowed_but_username_in_content_is_flagged(self):
        text = 'AVL Contributors <avl@users.noreply.github.com>\ncontact@example.com\n'
        self.assertEqual(scan_bytes('notice.txt', text.encode(), identities=())[0], [])
        findings, _ = scan_bytes('notice.txt', b'owner: private_owner', identities=('private_owner',))
        self.assertEqual(findings[0]['category'], 'local_identity')

    def test_binary_skip_is_explicit_and_paths_are_still_checked(self):
        findings, skipped = scan_bytes('weights.pt', b'PK\x00\xff', identities=())
        self.assertEqual(findings, [])
        self.assertEqual(skipped, {'file': 'weights.pt', 'bytes': 4, 'reason': 'binary_not_inspected'})
        findings, _ = scan_bytes('private_owner.pt', b'\x00', identities=('private_owner',))
        self.assertEqual(findings[0]['category'], 'local_identity')

    def test_git_index_bytes_are_inspected_and_untracked_files_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True)
            git('init', '-q')
            (root / 'source.txt').write_text('person' + '@mail.invalid')
            git('add', 'source.txt')
            (root / 'source.txt').write_text('clean working tree')
            (root / 'ignored.txt').write_text('sk-' + 'Y' * 40)
            report = inspect_git(root, 'staged', identities=())
            self.assertEqual(report['files_checked'], 1)
            self.assertEqual(report['findings'][0]['category'], 'email_address')
            git('config', 'user.name', 'Test')
            git('config', 'user.email', 'test@example.com')
            git('commit', '-qm', 'fixture')
            self.assertEqual(inspect_git(root, 'staged', identities=())['files_checked'], 0)
            self.assertEqual(inspect_git(root, 'tracked', identities=())['files_checked'], 1)

    def test_cli_failure_is_closed_and_never_echoes_git_exception(self):
        output = io.StringIO()
        with patch('antlab.publication_guard.inspect_git', side_effect=RuntimeError('private_owner')):
            with contextlib.redirect_stdout(output):
                self.assertEqual(main([]), 1)
        self.assertTrue(json.loads(output.getvalue())['blocked'])
        self.assertNotIn('private_owner', output.getvalue())

    def test_scanner_and_generated_test_fixtures_do_not_self_flag(self):
        import antlab.publication_guard as guard
        for filename in (Path(guard.__file__), Path(__file__)):
            findings, skipped = scan_bytes(filename.name, filename.read_bytes(), identities=())
            self.assertEqual(findings, [])
            self.assertIsNone(skipped)


if __name__ == '__main__':
    unittest.main()
