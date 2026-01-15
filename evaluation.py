import general
import cyk_process
import time
from datetime import datetime
import json

class CYKEvaluator:
    def __init__(self):
        self.results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'avg_parse_time': 0.0,
            'test_cases': []
        }
        
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        
        self.category_stats = {}
        self.pattern_stats = {}
    
    def load_dataset(self, filename="evaluation_dataset/evaluation_dataset.txt"):
        test_cases = []
        current_category = "General"
        
        print(f"\nLoading dataset from: {filename}")
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    if line.startswith('#'):
                        current_category = line.lstrip('#').strip()
                        if not current_category:
                            current_category = "General"
                        continue
                    
                    parts = line.split('|')
                    if len(parts) >= 2:
                        label = parts[0].strip()
                        sentence = parts[1].strip()
                        expected_pattern = parts[2].strip() if len(parts) >= 3 else None
                        
                        expected_valid = (label.upper() == 'VALID')
                        
                        test_cases.append({
                            'sentence': sentence,
                            'expected': expected_valid,
                            'expected_pattern': expected_pattern,
                            'category': current_category
                        })
            
            print(f"Loaded {len(test_cases)} test cases")
            return test_cases
            
        except FileNotFoundError:
            print(f"Error: File {filename} not found!")
            return []
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
    
    def normalize_pattern(self, pattern):
        """Normalize pattern untuk perbandingan"""
        if not pattern:
            return None

        pattern = pattern.replace('→', '').replace('->', '')
        pattern = ' '.join(pattern.split())
        return pattern.strip()
    
    def extract_components(self, pattern):
        """
        Extract komponen dari pattern.
        Contoh: 
        - "K → P S" → "P S"
        - "K → X1 Pel" → "X1 Pel"
        - "P S" → "P S" (already components)
        """
        if not pattern:
            return None
        
        pattern = self.normalize_pattern(pattern)
        
        # Check if contains "K" at the start (full derivation format)
        parts = pattern.split()
        if parts and parts[0] == 'K':
            # Remove "K" to get components only
            # "K P S" → "P S"
            # "K X1 Pel" → "X1 Pel"
            return ' '.join(parts[1:])
        
        # Already in component format
        return pattern
    
    def test_sentence(self, sentence, expected_valid, expected_pattern=None, category="General"):
        words = sentence.lower().split()
        
        start_time = time.time()
        is_known, unknown_words = general.check_alphabet(words)
        
        if not is_known:
            parse_time = time.time() - start_time
            result = {
                'sentence': sentence,
                'expected': expected_valid,
                'expected_pattern': expected_pattern,
                'expected_components': self.extract_components(expected_pattern),
                'actual': False,
                'actual_pattern': None,
                'actual_components': None,
                'correct': not expected_valid,
                'pattern_match': expected_pattern is None,
                'failure_reason': f"Unknown words: {', '.join(unknown_words)}",
                'parse_time': parse_time,
                'error': f"Unknown words: {', '.join(unknown_words)}",
                'category': category,
                'words': words,
                'parse_tree': None
            }
            self._update_metrics(result)
            return result
        
        try:
            table, backpointer = cyk_process.cyk_parse(words)
            n = len(words)
            is_valid = cyk_process.is_valid_sentence(table, n, "K")
            parse_time = time.time() - start_time
            
            parse_tree = None
            actual_pattern = None
            actual_components = None
            
            if is_valid:
                pattern_info = cyk_process.get_sentence_pattern(backpointer, words, "K")
                if pattern_info:
                    parse_tree = cyk_process.format_parse_tree(
                        pattern_info['parse_tree'], 
                        words, 
                        prefix=""
                    )
                    actual_pattern = pattern_info['pattern']
                    actual_components = self.extract_components(actual_pattern)
            
            pattern_match = True
            failure_reason = None
            expected_components = self.extract_components(expected_pattern)
            
            if expected_components:
                if actual_components:
                    pattern_match = expected_components == actual_components
                    
                    if not pattern_match:
                        failure_reason = f"Pattern mismatch: Expected '{expected_components}', Got '{actual_components}'"
                else:
                    pattern_match = False
                    failure_reason = "No pattern found (sentence rejected)"
            
            final_correct = is_valid == expected_valid
            if expected_components and not pattern_match:
                final_correct = False
            
            result = {
                'sentence': sentence,
                'expected': expected_valid,
                'expected_pattern': expected_pattern,
                'expected_components': expected_components,
                'actual': is_valid,
                'actual_pattern': actual_pattern,
                'actual_components': actual_components,
                'correct': final_correct,
                'pattern_match': pattern_match,
                'failure_reason': failure_reason,
                'parse_time': parse_time,
                'error': None,
                'category': category,
                'words': words,
                'parse_tree': parse_tree,
                'final_cell': str(cyk_process.get_parse_result(table, n))
            }
            
            self._update_metrics(result)
            return result
            
        except Exception as e:
            parse_time = time.time() - start_time
            result = {
                'sentence': sentence,
                'expected': expected_valid,
                'expected_pattern': expected_pattern,
                'expected_components': self.extract_components(expected_pattern),
                'actual': False,
                'actual_pattern': None,
                'actual_components': None,
                'correct': not expected_valid,
                'pattern_match': expected_pattern is None,
                'failure_reason': f"Exception: {str(e)}",
                'parse_time': parse_time,
                'error': str(e),
                'category': category,
                'words': words,
                'parse_tree': None
            }
            self._update_metrics(result)
            return result
    
    def _update_metrics(self, result):
        self.results['total_tests'] += 1
        self.results['test_cases'].append(result)
        
        if result['correct']:
            self.results['passed'] += 1
        else:
            self.results['failed'] += 1
        
        # Update confusion matrix
        if result['expected'] and result['actual'] and result['pattern_match']:
            self.true_positive += 1
        elif not result['expected'] and not result['actual']:
            self.true_negative += 1
        elif result['actual'] and (not result['expected'] or not result['pattern_match']):
            self.false_positive += 1
        elif result['expected'] and not result['actual']:
            self.false_negative += 1
        
        # Update category stats
        category = result['category']
        if category not in self.category_stats:
            self.category_stats[category] = {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'pattern_match': 0,
                'pattern_mismatch': 0
            }
        
        self.category_stats[category]['total'] += 1
        if result['correct']:
            self.category_stats[category]['passed'] += 1
        else:
            self.category_stats[category]['failed'] += 1
        
        if result['expected_pattern']:
            if result['pattern_match']:
                self.category_stats[category]['pattern_match'] += 1
            else:
                self.category_stats[category]['pattern_mismatch'] += 1
        
        if result['expected_components']:
            pattern = result['expected_components']
            if pattern not in self.pattern_stats:
                self.pattern_stats[pattern] = {
                    'total': 0,
                    'correct': 0,
                    'pattern_match': 0,
                    'pattern_mismatch': 0
                }
            
            self.pattern_stats[pattern]['total'] += 1
            if result['correct']:
                self.pattern_stats[pattern]['correct'] += 1
            if result['pattern_match']:
                self.pattern_stats[pattern]['pattern_match'] += 1
            else:
                self.pattern_stats[pattern]['pattern_mismatch'] += 1
    
    def calculate_final_metrics(self):
        total = self.results['total_tests']
        
        if total > 0:
            self.results['accuracy'] = (self.results['passed'] / total) * 100
            
            if (self.true_positive + self.false_positive) > 0:
                self.results['precision'] = (
                    self.true_positive / (self.true_positive + self.false_positive)
                ) * 100
            else:
                self.results['precision'] = 0
            
            if (self.true_positive + self.false_negative) > 0:
                self.results['recall'] = (
                    self.true_positive / (self.true_positive + self.false_negative)
                ) * 100
            else:
                self.results['recall'] = 0
            
            if self.results['precision'] + self.results['recall'] > 0:
                self.results['f1_score'] = (
                    2 * (self.results['precision'] * self.results['recall']) / 
                    (self.results['precision'] + self.results['recall'])
                )
            else:
                self.results['f1_score'] = 0
            
            total_time = sum(tc['parse_time'] for tc in self.results['test_cases'])
            self.results['avg_parse_time'] = total_time / total
    
    def print_summary(self):
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        print(f"\nOverall Statistics:")
        print(f"Total Tests: {self.results['total_tests']}")
        print(f"Passed: {self.results['passed']} ({(self.results['passed']/self.results['total_tests']*100):.1f}%)")
        print(f"Failed: {self.results['failed']} ({(self.results['failed']/self.results['total_tests']*100):.1f}%)")
        print(f"Accuracy: {self.results['accuracy']:.2f}%")
        
        print(f"\nConfusion Matrix:")
        print(f"┌─────────────────────┬──────────────────────┐")
        print(f"│                     │      Predicted       │")
        print(f"│       Actual        ├──────────┬───────────┤")
        print(f"│                     │  Valid   │  Invalid  │")
        print(f"├─────────────────────┼──────────┼───────────┤")
        print(f"│      Valid          │   {self.true_positive:3d}    │    {self.false_negative:3d}    │")
        print(f"│     Invalid         │   {self.false_positive:3d}    │    {self.true_negative:3d}    │")
        print(f"└─────────────────────┴──────────┴───────────┘")
        print(f"\nNote: Pattern validation checks COMPONENTS only (e.g., 'P S' not 'K → P S')")
        
        print(f"\nClassification Metrics:")
        print(f"Precision: {self.results['precision']:.2f}%")
        print(f"Recall:    {self.results['recall']:.2f}%")
        print(f"F1 Score:  {self.results['f1_score']:.2f}%")
        
        print(f"\nPerformance Metrics:")
        print(f"Average Parse Time: {self.results['avg_parse_time']*1000:.2f}ms")
        print(f"Total Processing Time: {sum(tc['parse_time'] for tc in self.results['test_cases']):.2f}s")
        
        if self.pattern_stats:
            print(f"\nPattern Accuracy (by Components):")
            print(f"{'Pattern':<30} {'Total':>6} {'Match':>6} {'Mismatch':>6} {'Acc%':>6}")
            print("-" * 70)
            for pattern, stats in sorted(self.pattern_stats.items()):
                acc = (stats['pattern_match'] / stats['total'] * 100) if stats['total'] > 0 else 0
                print(f"{pattern:<30} {stats['total']:>6} {stats['pattern_match']:>6} {stats['pattern_mismatch']:>6} {acc:>5.1f}%")
        
        if self.category_stats:
            print(f"\nCategory Breakdown:")
            print(f"{'Category':<35} {'Total':>6} {'Pass':>6} {'Fail':>6} {'PMat':>6} {'PMis':>6} {'Acc%':>6}")
            print("-" * 70)
            for category, stats in sorted(self.category_stats.items()):
                acc = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
                cat_display = category[:34] if len(category) > 34 else category
                pmat = stats.get('pattern_match', 0)
                pmis = stats.get('pattern_mismatch', 0)
                print(f"{cat_display:<35} {stats['total']:>6} {stats['passed']:>6} {stats['failed']:>6} {pmat:>6} {pmis:>6} {acc:>5.1f}%")
        
        print("\n" + "="*70)
    
    def print_failed_cases(self):
        failed_cases = [tc for tc in self.results['test_cases'] if not tc['correct']]
        
        if not failed_cases:
            print("\n✅ All test cases passed!")
            return
        
        print(f"\n❌ Failed Test Cases ({len(failed_cases)} cases):")
        print("="*70)
        
        for idx, tc in enumerate(failed_cases, 1):
            print(f"\n{idx}. {tc['sentence']}")
            print(f"   Expected: {'VALID' if tc['expected'] else 'INVALID'}")
            print(f"   Actual:   {'VALID' if tc['actual'] else 'INVALID'}")
            print(f"   Category: {tc['category']}")
            
            if tc.get('expected_components'):
                print(f"   Expected Components: {tc['expected_components']}")
                print(f"   Actual Components:   {tc.get('actual_components', 'None')}")
                print(f"   Pattern Match:       {'✅' if tc['pattern_match'] else '❌'}")
            
            if tc.get('failure_reason'):
                print(f"   Failure Reason: {tc['failure_reason']}")
            elif tc['error']:
                print(f"   Error: {tc['error']}")
    
    def print_pattern_mismatch_cases(self):
        """Print cases dimana pola tidak match dengan expected"""
        mismatch_cases = [tc for tc in self.results['test_cases'] 
                         if tc.get('expected_components') and not tc['pattern_match']]
        
        if not mismatch_cases:
            print("\n✅ All patterns matched!")
            return
        
        print(f"\nPattern Mismatch Cases ({len(mismatch_cases)} cases):")
        print("="*70)
        print("Note: Pattern validation checks COMPONENTS only")
        print("="*70)
        
        for idx, tc in enumerate(mismatch_cases, 1):
            print(f"\n{idx}. {tc['sentence']}")
            print(f"   Expected Components: {tc['expected_components']}")
            print(f"   Actual Components:   {tc.get('actual_components', 'None')}")
            if tc.get('actual_pattern'):
                print(f"   Actual Full Pattern: {tc['actual_pattern']}")
            print(f"   Category: {tc['category']}")
            print(f"   Parser Result: {'VALID' if tc['actual'] else 'INVALID'}")
            print(f"   Final Result:  ❌ FAILED (component mismatch)")
    
    def save_report(self, filename="evaluation_report.json"):
        report = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_mode': 'component_pattern_validation',
            'note': 'Pattern validation checks COMPONENTS only (e.g., P S not K → P S)',
            'summary': {
                'total_tests': self.results['total_tests'],
                'passed': self.results['passed'],
                'failed': self.results['failed'],
                'accuracy': self.results['accuracy'],
                'precision': self.results['precision'],
                'recall': self.results['recall'],
                'f1_score': self.results['f1_score'],
                'avg_parse_time': self.results['avg_parse_time']
            },
            'confusion_matrix': {
                'true_positive': self.true_positive,
                'true_negative': self.true_negative,
                'false_positive': self.false_positive,
                'false_negative': self.false_negative
            },
            'pattern_stats': self.pattern_stats,
            'category_stats': self.category_stats,
            'test_cases': [
                {
                    'sentence': tc['sentence'],
                    'expected': tc['expected'],
                    'expected_pattern': tc.get('expected_pattern'),
                    'expected_components': tc.get('expected_components'),
                    'actual': tc['actual'],
                    'actual_pattern': tc.get('actual_pattern'),
                    'actual_components': tc.get('actual_components'),
                    'correct': tc['correct'],
                    'pattern_match': tc.get('pattern_match'),
                    'failure_reason': tc.get('failure_reason'),
                    'parse_time': tc['parse_time'],
                    'category': tc['category'],
                    'error': tc['error']
                }
                for tc in self.results['test_cases']
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nReport saved to: {filename}")


def run_evaluation(dataset_file="evaluation_dataset/evaluation_dataset.txt"):
    evaluator = CYKEvaluator()
    
    print("\n" + "="*70)
    print("SEKEN App - Evaluation (Component Pattern Validation)")
    print("Sistem Parsing Kalimat Bahasa Bali Berpredikat Frasa Presisi")
    print("dengan algoritma CYK")
    print("="*70)
    
    test_cases = evaluator.load_dataset(dataset_file)
    
    if not test_cases:
        print("No test cases loaded. Exiting.")
        return evaluator
    
    print(f"\nRunning {len(test_cases)} test cases...")
    print("-" * 70)
    
    for idx, tc in enumerate(test_cases, 1):
        print(f"\n[{idx}/{len(test_cases)}] Testing: {tc['sentence']}")
        
        result = evaluator.test_sentence(
            sentence=tc['sentence'],
            expected_valid=tc['expected'],
            expected_pattern=tc.get('expected_pattern'),
            category=tc['category']
        )
        
        status = "✅ PASS" if result['correct'] else "❌ FAIL"
        expected_str = "VALID" if result['expected'] else "INVALID"
        actual_str = "VALID" if result['actual'] else "INVALID"
        print(f"   {status}: Expected={expected_str}, Actual={actual_str}")
        
        if result.get('expected_components'):
            if result['pattern_match']:
                print(f"   Components ✅: {result['expected_components']}")
            else:
                print(f"   Components ❌: Expected='{result['expected_components']}', Got='{result.get('actual_components', 'None')}'")
        elif result.get('actual_components'):
            print(f"   Components: {result['actual_components']}")
    
    evaluator.calculate_final_metrics()
    evaluator.print_summary()
    evaluator.print_failed_cases()
    evaluator.print_pattern_mismatch_cases()
    evaluator.save_report("evaluation_report.json")
    
    return evaluator


if __name__ == "__main__":
    import sys
    
    dataset_file = "evaluation_dataset/evaluation_dataset.txt"
    
    if len(sys.argv) > 1:
        dataset_file = sys.argv[1]
    
    evaluator = run_evaluation(dataset_file)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETED")
    print("="*70)
    print(f"\nSummary:")
    print(f"  - Total Tests: {evaluator.results['total_tests']}")
    print(f"  - Accuracy: {evaluator.results['accuracy']:.2f}%")
    print(f"  - F1 Score: {evaluator.results['f1_score']:.2f}%")
    print(f"\nReport saved to: evaluation_report.json")
    print("="*70)