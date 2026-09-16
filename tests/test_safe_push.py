import re
import unittest

from scripts.safe_push import scan_text


class SafePushScannerTests(unittest.TestCase):
    def test_detects_private_key(self):
        value = "-----BEGIN " + "PRIVATE KEY-----"
        findings = scan_text("config.txt", value)
        self.assertTrue(any(item.severity == "hard" for item in findings))

    def test_detects_confidential_marker(self):
        findings = scan_text("notes.md", "仅供内部使用，不得外传")
        self.assertTrue(any(item.severity == "review" for item in findings))

    def test_ignores_example_credentials(self):
        findings = scan_text("docs.md", 'api_key = "example-placeholder-value"')
        self.assertFalse(any(item.severity == "hard" for item in findings))

    def test_applies_local_patterns_without_echoing_match(self):
        findings = scan_text("notes.md", "Project Nebula", [("local", re.compile("nebula", re.I))])
        self.assertEqual(findings[0].location, "notes.md")
        self.assertEqual(findings[0].rule, "local")


if __name__ == "__main__":
    unittest.main()
