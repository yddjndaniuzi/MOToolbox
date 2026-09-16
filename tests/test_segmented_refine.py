from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pressconf import model_client
from pressconf.brief import parse_transcript, segment_cues
from pressconf.refine import refine_brief, extract_section_images
from pressconf.segmented_refine import (
    MAX_EVIDENCE_CHARS, parse_batch, plan_batches, refine_segmented, section_ids, validate_sections,
)


def fixtures(count=9):
    source = '\n\n'.join(f'**{i}. 原标题**\n- 要点\n  ![](frames/{i}.jpg)' for i in range(1, count + 1))
    transcript = '\n\n'.join(
        f'{i+1}\n00:{i*3:02d}:00,000 --> 00:{i*3:02d}:01,000\n产品{i+1} 售价{i+1}999元 片段尾证据{i+1}'
        for i in range(count))
    return source, transcript


class SegmentedTests(unittest.TestCase):
    def test_every_segment_consumed_and_bounded(self):
        source, transcript = fixtures()
        batches = plan_batches(source, transcript)
        self.assertEqual([len(batch) for batch in batches], [4, 4, 1])
        units = [unit for batch in batches for unit in batch]
        expected = segment_cues(parse_transcript(transcript))
        self.assertEqual([unit['text'] for unit in units], [item.text for item in expected])
        self.assertTrue(all(sum(len(unit['text']) for unit in batch) <= MAX_EVIDENCE_CHARS for batch in batches))
        self.assertIn('片段尾证据9', units[-1]['text'])
        self.assertIn('片段尾证据4', batches[1][0]['context_before'])

    def test_untimed_oversized_chapter_is_not_truncated(self):
        text = '开始' + '长材料' * 15000 + '尾证据'
        batches = plan_batches('**1. 章节**', text)
        units = [unit for batch in batches for unit in batch]
        self.assertGreater(len(units), 1)
        self.assertEqual(''.join(unit['text'] for unit in units), text)
        self.assertTrue(all(len(unit['text']) <= MAX_EVIDENCE_CHARS for unit in units))

    def test_alignment_and_final_validation_fail_closed(self):
        source, transcript = fixtures()
        with self.assertRaises(RuntimeError):
            plan_batches(source, '')
        for result in ['**1. 只有开头**', source + '\n**9. 重复**', source.replace('**2.', '**3.')]:
            with self.assertRaises(RuntimeError):
                validate_sections(source, result)

    def test_markdown_transport_preserves_quotes_and_linebreaks(self):
        source, transcript = fixtures(1)
        batch = plan_batches(source, transcript)[0]
        text = '<!-- FACTS -->\n官方说"售价1999元"。\n<!-- DETAILS -->\n**1. 新品价格**\n- 官方说"售价1999元"。\n- 条件：首发。'
        data = parse_batch(text, batch)
        self.assertEqual(data['sections'][0]['body'], '- 官方说"售价1999元"。\n- 条件：首发。')
        with self.assertRaises(ValueError):
            parse_batch(text.replace('**1.', '**2.'), batch)

    def test_resume_completed_batches_and_keep_formal_on_failure(self):
        source, transcript = fixtures()
        fail_summary = True
        calls = []
        def fake_model(**kwargs):
            prompt = kwargs['messages'][-1]['content']
            kwargs['on_metadata']({'stop_reason': 'end_turn', 'usage': {'output_tokens': 100}})
            if '<batch>' in prompt:
                batch = json.loads(prompt.split('<batch>')[1].split('</batch>')[0])
                calls.append([item['number'] for item in batch])
                result = json.dumps({'facts': '产品、价格及末段证据', 'sections': [
                    {'number': item['number'], 'title': '产品发布', 'body': item['time'] + '\n- ' + item['text']}
                    for item in batch]}, ensure_ascii=False)
                kwargs['on_delta'](result)
                return result
            calls.append('summary')
            if fail_summary:
                raise model_client.IncompleteGenerationError('max_tokens')
            return '# 发布会\n\n## 发布会概述\n\n全部产品已发布。'
        with tempfile.TemporaryDirectory() as temp, patch('pressconf.segmented_refine.stream_chat_model', side_effect=fake_model):
            root = Path(temp)
            formal = root / 'brief_refined.md'
            formal.write_text('原有正式稿')
            config = dict(provider='anthropic', base_url='https://test.invalid', model='test', api_key='secret')
            kwargs = dict(result_dir=root, source=source, transcript=transcript,
                          instructions='完整覆盖', model_config=config, task_type='business_review')
            with self.assertRaises(model_client.IncompleteGenerationError):
                refine_segmented(**kwargs)
            self.assertEqual(formal.read_text(), '原有正式稿')
            self.assertEqual(section_ids((root / 'brief_refined.partial.md').read_text()), list(range(1, 10)))
            fail_summary = False
            calls.clear()
            result, state = refine_segmented(**kwargs)
            self.assertEqual(calls, ['summary'])
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(section_ids(result), list(range(1, 10)))
            self.assertIn('片段尾证据9', result)
            self.assertNotIn('secret', json.dumps(state))
            calls.clear()
            refine_segmented(**{**kwargs, 'instructions': '新要求'})
            self.assertEqual(len(calls), 4)

    def test_refine_pipeline_publishes_only_complete_output_with_images(self):
        source, transcript = fixtures()
        def fake_model(**kwargs):
            prompt = kwargs['messages'][-1]['content']
            kwargs['on_metadata']({'stop_reason': 'end_turn', 'usage': {'output_tokens': 100}})
            if '<batch>' in prompt:
                batch = json.loads(prompt.split('<batch>')[1].split('</batch>')[0])
                return '<!-- FACTS -->\n全部产品价格\n<!-- DETAILS -->\n' + '\n\n'.join(
                    f'**{item["number"]}. 产品发布**\n- {item["text"]}' for item in batch)
            return '# 发布会\n\n## 发布会概述\n\n全部产品已发布。'
        with tempfile.TemporaryDirectory() as temp, patch('pressconf.segmented_refine.stream_chat_model', side_effect=fake_model):
            root = Path(temp)
            (root / 'transcript').mkdir()
            (root / 'transcript/asr.srt').write_text(transcript)
            (root / 'transcript/meta.json').write_text(json.dumps({'path': 'transcript/asr.srt'}))
            (root / 'brief_base.md').write_text(source)
            config = dict(provider='anthropic', base_url='https://test.invalid', model='test', api_key='secret')
            output, meta = refine_brief(root, '发布会', root, model_config=config)
            self.assertEqual(section_ids(output.read_text()), list(range(1, 10)))
            self.assertEqual(extract_section_images(output.read_text()), extract_section_images(source))
            self.assertEqual(meta['evidence']['coverage_ratio'], 1.0)
            self.assertFalse((root / 'brief_refined.partial.md').exists())
            self.assertEqual(meta['generation']['status'], 'complete')
            original = output.read_text()
            with patch('pressconf.segmented_refine.stream_chat_model', side_effect=model_client.IncompleteGenerationError('max_tokens')):
                with self.assertRaises(model_client.IncompleteGenerationError):
                    refine_brief(root, '发布会', root, user_instruction='新要求', model_config=config)
            self.assertEqual(output.read_text(), original)


class CompletionTests(unittest.TestCase):
    def stream(self, provider, events):
        body = b''.join(('data: ' + (event if isinstance(event, str) else json.dumps(event)) + '\n\n').encode() for event in events)
        metadata = {}
        with patch('pressconf.model_client.urllib.request.urlopen', return_value=io.BytesIO(body)):
            result = model_client.stream_chat_model(
                provider=provider, api_key='test', base_url='https://test.invalid', model='test',
                messages=[], on_metadata=metadata.update,
            )
        return result, metadata

    def test_anthropic_success_records_usage(self):
        result, meta = self.stream('anthropic', [
            {'type': 'message_start', 'message': {'usage': {'input_tokens': 42}}},
            {'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': '完整'}},
            {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 2}},
            {'type': 'message_stop'},
        ])
        self.assertEqual(result, '完整')
        self.assertEqual(meta['usage'], {'input_tokens': 42, 'output_tokens': 2})

    def test_truncated_and_missing_terminal_rejected(self):
        for reason, terminal in [('max_tokens', True), ('end_turn', False), (None, True)]:
            events = [{'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': '半句话'}},
                      {'type': 'message_delta', 'delta': {'stop_reason': reason}}]
            if terminal:
                events.append({'type': 'message_stop'})
            with self.assertRaises(model_client.IncompleteGenerationError):
                self.stream('anthropic', events)
        for reason in ['length', None, 'content_filter']:
            with self.assertRaises(model_client.IncompleteGenerationError):
                self.stream('openai', [{'choices': [{'delta': {'content': '半句话'}, 'finish_reason': reason}]}, '[DONE]'])

    def test_openai_success(self):
        result, meta = self.stream('openai', [
            {'choices': [{'delta': {'content': '完整'}, 'finish_reason': None}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 2}}, '[DONE]',
        ])
        self.assertEqual(result, '完整')
        self.assertEqual(meta['stop_reason'], 'stop')

    def test_nonstream_limits_rejected(self):
        for provider, data in [('openai', {'choices': [{'message': {'content': '半句'}, 'finish_reason': 'length'}]}),
                               ('anthropic', {'content': [{'type': 'text', 'text': '半句'}], 'stop_reason': 'max_tokens'})]:
            with patch('pressconf.model_client.post_json', return_value=data), self.assertRaises(model_client.IncompleteGenerationError):
                model_client.call_chat_model(provider=provider, api_key='test', base_url='https://test.invalid', model='test', messages=[])


if __name__ == '__main__':
    unittest.main()
