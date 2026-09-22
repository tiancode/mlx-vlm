"""Context-boundary probes follow the running service configuration."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics.check_long_service import remaining_context_budget


class ContextBudgetTests(unittest.TestCase):
    def test_uses_effective_limit_instead_of_model_capacity(self):
        health = {"loaded_context_size": 1048576, "effective_context_limit": 393216}
        self.assertEqual(remaining_context_budget(health, 8192), 385024)
        health["effective_context_limit"] = 262144
        self.assertEqual(remaining_context_budget(health, 8192), 253952)

    def test_rejects_missing_or_unbounded_limit(self):
        for limit in (None, 0, -1):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "effective_context_limit"):
                remaining_context_budget({"effective_context_limit": limit}, 64)

    def test_requires_room_for_generation(self):
        for prompt_tokens in (-1, 4096, 4097):
            with self.subTest(prompt_tokens=prompt_tokens), self.assertRaisesRegex(ValueError, "generation budget"):
                remaining_context_budget({"effective_context_limit": 4096}, prompt_tokens)


if __name__ == "__main__":
    unittest.main()
