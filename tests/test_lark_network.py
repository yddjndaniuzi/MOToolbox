import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

from pressconf import lark_export as lark


class LarkNetworkTests(unittest.TestCase):
    def test_upload_retries_connection_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'a.jpg'
            image.write_bytes(b'image')
            response = MagicMock()
            response.__enter__.return_value.read.return_value = b'{"success":true,"data":{"url":"https://example.com/a.jpg"}}'
            with patch.object(lark.urllib.request, 'urlopen', side_effect=[ConnectionResetError(54, 'reset'), response]) as request, patch.object(lark.time, 'sleep'):
                self.assertEqual(lark.upload_imgbb(image, api_key='test'), 'https://example.com/a.jpg')
                self.assertEqual(request.call_count, 2)
            with patch.object(lark.urllib.request, 'urlopen', side_effect=ConnectionResetError(54, 'reset')) as request, patch.object(lark.time, 'sleep'):
                with self.assertRaisesRegex(RuntimeError, '已尝试 3 次'):
                    lark.upload_imgbb(image, api_key='test')
                self.assertEqual(request.call_count, 3)

    def test_partial_upload_is_saved_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'frames').mkdir()
            for name in ['a.jpg', 'b.jpg']:
                (root / 'frames' / name).write_bytes(b'image')
            markdown = '![](frames/a.jpg)\n![](frames/b.jpg)'
            config = {'image_upload_provider': 'imgbb', 'imgbb_api_key': 'test'}
            with patch.object(lark, 'upload_imgbb', side_effect=['https://example.com/a.jpg', RuntimeError('offline')]):
                with self.assertRaisesRegex(RuntimeError, 'frames/b.jpg'):
                    lark.resolve_image_urls(markdown, root, config)
            self.assertIn('frames/a.jpg', json.loads((root / 'image_uploads.json').read_text())['uploads'])
            with patch.object(lark, 'upload_imgbb', return_value='https://example.com/b.jpg') as upload:
                self.assertEqual(len(lark.resolve_image_urls(markdown, root, config)), 2)
                upload.assert_called_once()

    def test_create_is_not_retried_after_reset(self):
        with patch.object(lark.urllib.request, 'urlopen', side_effect=ConnectionResetError(54, 'reset')) as request:
            with self.assertRaisesRegex(RuntimeError, '避免重复创建'):
                lark.post_json('https://example.com', {}, {})
            request.assert_called_once()

    def test_mcp_poll_keeps_original_task_id_until_document_url_is_ready(self):
        first_response = {
            'result': {
                'content': [{
                    'type': 'text',
                    'text': json.dumps({
                        'status': 'running',
                        'task_id': 'task-123',
                    }),
                }],
            },
        }
        still_running_without_task_id = {
            'result': {
                'content': [{
                    'type': 'text',
                    'text': json.dumps({
                        'status': 'running',
                        'message': '任务仍在处理中',
                    }),
                }],
            },
        }
        completed = {
            'result': {
                'content': [{
                    'type': 'text',
                    'text': json.dumps({
                        'status': 'success',
                        'document_id': 'docx-test',
                        'url': 'https://example.feishu.cn/docx/docx-test',
                    }),
                }],
            },
        }
        original_payload = {
            'jsonrpc': '2.0',
            'id': 'motoolbox-lark-export',
            'method': 'tools/call',
            'params': {'name': 'create-doc', 'arguments': {'markdown': '# test'}},
        }

        with patch.object(
            lark,
            'post_json',
            side_effect=[still_running_without_task_id, completed],
        ) as post, patch.object(lark.time, 'sleep'):
            response, attempts = lark.resolve_mcp_task_result(
                'https://example.com/mcp',
                original_payload,
                {},
                first_response,
            )

        self.assertEqual(response, completed)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(post.call_count, 2)
        for call in post.call_args_list:
            self.assertEqual(call.args[1]['params']['arguments'], {'task_id': 'task-123'})
        fields = lark.extract_lark_export_fields(lark.normalize_mcp_response(response))
        self.assertEqual(fields['url'], 'https://example.feishu.cn/docx/docx-test')
