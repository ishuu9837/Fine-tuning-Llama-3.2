import ast
from collections import Counter
import json
from pathlib import Path
import unittest

NOTEBOOK = next(Path(__file__).resolve().parents[1].glob('*.ipynb'))

class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text())
        cls.code = '\n'.join(''.join(c['source']) for c in cls.notebook['cells'] if c['cell_type'] == 'code')
        # Execute only the actual pure scoring functions, without GPU dependencies.
        source = ''.join(cls.notebook['cells'][16]['source'])
        functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                     and n.name in ('normalize_answer', 'answer_scores')]
        cls.scope = {'Counter': Counter}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), 'exec'), cls.scope)

    def test_exact_and_paraphrase(self):
        score = self.scope['answer_scores']([' A  bond ', 'net profit'], ['a bond', 'profit'])
        self.assertEqual(score['exact_match_accuracy_pct'], 50)
        self.assertAlmostEqual(score['answer_token_f1_pct'], 100 * (1 + 2/3) / 2)

    def test_financial_numbers_are_distinct(self):
        for a, b in [('1.5', '15'), ('-5', '5'), ('$5', '5')]:
            self.assertEqual(self.scope['answer_scores']([a], [b])['exact_match_accuracy_pct'], 0)

    def test_empty_and_mismatched_inputs(self):
        for a, b in [([], []), (['a'], ['a', 'b'])]:
            with self.assertRaises(ValueError):
                self.scope['answer_scores'](a, b)
        self.assertEqual(self.scope['answer_scores']([''], ['value'])['answer_token_f1_pct'], 0)

    def test_repeated_words_do_not_inflate_f1(self):
        self.assertEqual(self.scope['answer_scores'](['profit profit profit'], ['profit'])['answer_token_f1_pct'], 50)

    def test_code_syntax_and_cleared_outputs(self):
        for cell in self.notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            clean = '\n'.join(line for line in source.splitlines() if not line.startswith(('!', '%')))
            ast.parse(clean)
            self.assertEqual(cell['outputs'], [])
            self.assertIsNone(cell['execution_count'])

if __name__ == '__main__':
    unittest.main()
