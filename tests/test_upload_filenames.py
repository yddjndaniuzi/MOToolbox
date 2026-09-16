import io
import tempfile
import unittest
from pathlib import Path

from werkzeug.datastructures import FileStorage

from pressconf.media_feedback import extract_feedback_source
from pressconf.web import (
    save_content_review_lab_upload,
    save_derivative_upload,
    save_feedback_upload,
)


class UploadFilenameTests(unittest.TestCase):
    def test_document_uploads_preserve_extensions_and_readable_content(self):
        for saver in (
            lambda upload, root: save_feedback_upload(upload, root, "questionnaire"),
            lambda upload, root: save_derivative_upload(upload, root, "reference"),
            save_content_review_lab_upload,
        ):
            for name in ("媒体问卷.txt", "反馈.TXT", "产品 review.txt", "../../问卷.txt"):
                with self.subTest(saver=saver, name=name), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    upload = FileStorage(stream=io.BytesIO("媒体反馈正文".encode()), filename=name)
                    saved = saver(upload, root)
                    self.assertEqual(saved.parent, root)
                    self.assertEqual(saved.suffix, ".txt")
                    self.assertIn("媒体反馈正文", extract_feedback_source(saved))

    def test_chinese_spreadsheet_keeps_xlsx_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            upload = FileStorage(stream=io.BytesIO(b"xlsx-content"), filename="媒体问卷.xlsx")
            saved = save_feedback_upload(upload, Path(temp), "questionnaire")
            self.assertEqual(saved.suffix, ".xlsx")
            self.assertEqual(saved.read_bytes(), b"xlsx-content")

    def test_unsupported_extension_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            upload = FileStorage(stream=io.BytesIO(b"content"), filename="问卷.exe")
            with self.assertRaises(ValueError):
                save_feedback_upload(upload, Path(temp), "questionnaire")
            self.assertEqual(list(Path(temp).iterdir()), [])
