import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pressconf import runtime, ytdlp_update as updater
from pressconf.web import create_app


class YtdlpUpdateTests(unittest.TestCase):
    def test_verified_install_and_command_selection(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(updater, 'data_root', return_value=Path(tmp)), patch.object(runtime, 'data_root', return_value=Path(tmp)), patch.object(updater.sys, 'platform', 'darwin'):
            payload = b'new binary'
            downloads = [json.dumps({'tag_name': '2026.08.19'}).encode(), (hashlib.sha256(payload).hexdigest() + '  yt-dlp_macos\n').encode(), payload]
            with patch.object(updater, 'fetch', side_effect=downloads), patch.object(updater, 'version', return_value='2026.08.19'):
                self.assertEqual(updater.install_latest(), '2026.08.19')
            self.assertEqual(runtime.ytdlp_command('--version'), [str(Path(tmp) / 'tools/yt-dlp'), '--version'])
            self.assertEqual(updater.managed_binary().read_bytes(), payload)

    def test_failed_validation_preserves_previous_binary(self):
        for bad_checksum in [True, False]:
            with self.subTest(bad_checksum=bad_checksum), tempfile.TemporaryDirectory() as tmp, patch.object(updater, 'data_root', return_value=Path(tmp)), patch.object(updater.sys, 'platform', 'darwin'):
                target = updater.managed_binary()
                target.parent.mkdir()
                target.write_bytes(b'old')
                checksum = '0' * 64 if bad_checksum else hashlib.sha256(b'new').hexdigest()
                downloads = [b'{"tag_name":"2026.08.19"}', (checksum + '  yt-dlp_macos\n').encode(), b'new']
                with patch.object(updater, 'fetch', side_effect=downloads), patch.object(updater, 'version', side_effect=RuntimeError('cannot execute')):
                    with self.assertRaises(RuntimeError):
                        updater.install_latest()
                self.assertEqual(target.read_bytes(), b'old')
                self.assertEqual(list(target.parent.iterdir()), [target])

    def test_update_api_and_security(self):
        client = create_app().test_client()
        with patch.object(updater, 'start_update', return_value=True) as start, patch('pressconf.web.sys.platform', 'darwin'):
            self.assertEqual(client.post('/api/ytdlp/update', json={}).status_code, 202)
            start.return_value = False
            self.assertEqual(client.post('/api/ytdlp/update', json={}).status_code, 409)
            start.reset_mock()
            self.assertEqual(client.post('/api/ytdlp/update', json={}, headers={'Origin': 'https://example.com'}).status_code, 403)
            self.assertEqual(client.post('/api/ytdlp/update', json={}, environ_overrides={'REMOTE_ADDR': '10.0.0.1'}).status_code, 403)
            self.assertEqual(client.post('/api/ytdlp/update', data={}).status_code, 415)
            start.assert_not_called()
        self.assertIn('更新至最新稳定版', client.get('/admin').get_data(as_text=True))

    def test_duplicate_update_does_not_spawn_thread(self):
        with patch.dict(updater._STATE, running=True), patch.object(updater.threading, 'Thread') as thread:
            self.assertFalse(updater.start_update())
            thread.assert_not_called()
