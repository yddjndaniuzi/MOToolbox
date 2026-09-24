import unittest

from pressconf.model_client import anthropic_payload, openai_compatible_payload


class ModelClientTests(unittest.TestCase):
    def test_openai_payload_omits_unsupported_temperature(self):
        payload = openai_compatible_payload(
            model="azure_openai/gpt-5.6-sol",
            messages=[{"role": "user", "content": "User"}],
            temperature=0.35,
            max_tokens=100,
            stream=True,
        )

        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["model"], "azure_openai/gpt-5.6-sol")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "User"}])
        self.assertEqual(payload["max_tokens"], 100)
        self.assertTrue(payload["stream"])

    def test_anthropic_payload_omits_deprecated_temperature(self):
        payload = anthropic_payload(
            model="ppio/pa/claude-opus-4-8",
            messages=[
                {"role": "system", "content": "System"},
                {"role": "user", "content": "User"},
            ],
            temperature=0.2,
            max_tokens=100,
            stream=True,
        )

        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["model"], "ppio/pa/claude-opus-4-8")
        self.assertEqual(payload["system"], "System")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "User"}])
        self.assertEqual(payload["max_tokens"], 100)
        self.assertTrue(payload["stream"])


if __name__ == "__main__":
    unittest.main()
