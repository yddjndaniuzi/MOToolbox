import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pressconf.web import create_app


class StorageOpenTests(unittest.TestCase):
    def test_open_and_reject_unsafe_requests(self):
        with tempfile.TemporaryDirectory() as temp, patch('pressconf.web.RAW_ROOT', Path(temp)), patch('pressconf.web.sys.platform', 'darwin'), patch('pressconf.web.subprocess.run') as run:
            (Path(temp) / '发布会' / 'frames').mkdir(parents=True)
            client = create_app().test_client()
            response = client.post('/api/storage/open', json={'location': 'raw', 'slug': '发布会'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(run.call_args.args[0], ['/usr/bin/open', str(Path(temp).resolve() / '发布会')])
            run.reset_mock()
            for payload in [{'location': 'raw', 'slug': '..'}, {'location': 'raw', 'slug': '/tmp'}, {'location': 'arbitrary'}, {'location': []}]:
                self.assertEqual(client.post('/api/storage/open', json=payload).status_code, 400)
            self.assertEqual(client.post('/api/storage/open', json={'location': 'raw'}, headers={'Origin': 'https://other.example'}).status_code, 403)
            self.assertEqual(client.post('/api/storage/open', json={'location': 'raw'}, environ_overrides={'REMOTE_ADDR': '192.168.1.10'}).status_code, 403)
            self.assertEqual(client.post('/api/storage/open', data={'location': 'raw'}).status_code, 415)
            self.assertEqual(client.post('/api/storage/open', json={'location': 'raw', 'slug': 'missing'}).status_code, 404)
            run.assert_not_called()

    def test_folder_pages_render(self):
        client = create_app().test_client()
        for route in ['/', '/admin']:
            response = client.get(route)
            self.assertEqual(response.status_code, 200)
            self.assertIn('文件与缓存', response.get_data(as_text=True))
