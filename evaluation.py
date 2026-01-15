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
                        
                        expected_valid = (label.upper() == 'VALID')
                        
                        test_cases.append({
                            'sentence': sentence,
                            'expected': expected_valid,
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
    
    def test_sentence(self, sentence, expected_valid, category="General"):
        words = sentence.lower().split()
        
        start_time = time.time()
        is_known, unknown_words = general.check_alphabet(words)
        
        if not is_known:
            parse_time = time.time() - start_time
            result = {
                'sentence': sentence,
                'expected': expected_valid,
                'actual': False,
                'correct': not expected_valid,
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
            pattern = None
            if is_valid:
                pattern_info = cyk_process.get_sentence_pattern(backpointer, words, "K")
                if pattern_info:
                    parse_tree = cyk_process.format_parse_tree(
                        pattern_info['parse_tree'], 
                        words, 
                        prefix=""
                    )
                    pattern = pattern_info['pattern']
            result = {
                'sentence': sentence,
                'expected': expected_valid,
                'actual': is_valid,
                'correct': is_valid == expected_valid,
                'parse_time': parse_time,
                'error': None,
                'category': category,
                'words': words,
                'parse_tree': parse_tree,
                'pattern': pattern,
                'final_cell': str(cyk_process.get_parse_result(table, n))
            }
            
            
            self._update_metrics(result)
            return result
            
        except Exception as e:
            parse_time = time.time() - start_time
            result = {
                'sentence': sentence,
                'expected': expected_valid,
                'actual': False,
                'correct': not expected_valid,
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
        if result['expected'] and result['actual']:
            self.true_positive += 1
        elif not result['expected'] and not result['actual']:
            self.true_negative += 1
        elif not result['expected'] and result['actual']:
            self.false_positive += 1
        elif result['expected'] and not result['actual']:
            self.false_negative += 1
        
        category = result['category']
        if category not in self.category_stats:
            self.category_stats[category] = {
                'total': 0,
                'passed': 0,
                'failed': 0
            }
        
        self.category_stats[category]['total'] += 1
        if result['correct']:
            self.category_stats[category]['passed'] += 1
        else:
            self.category_stats[category]['failed'] += 1
    
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
        
        print(f"\nClassification Metrics:")
        print(f"Precision: {self.results['precision']:.2f}%")
        print(f"Recall:    {self.results['recall']:.2f}%")
        print(f"F1 Score:  {self.results['f1_score']:.2f}%")
        
        print(f"\nPerformance Metrics:")
        print(f"Average Parse Time: {self.results['avg_parse_time']*1000:.2f}ms")
        print(f"Total Processing Time: {sum(tc['parse_time'] for tc in self.results['test_cases']):.2f}s")
        
        if self.category_stats:
            print(f"\nCategory Breakdown:")
            print(f"{'Category':<45} {'Total':>6} {'Pass':>6} {'Fail':>6} {'Acc%':>6}")
            print("-" * 70)
            for category, stats in sorted(self.category_stats.items()):
                acc = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
                cat_display = category[:44] if len(category) > 44 else category
                print(f"{cat_display:<45} {stats['total']:>6} {stats['passed']:>6} {stats['failed']:>6} {acc:>5.1f}%")
        
        print("\n" + "="*70)
    
    def print_failed_cases(self):
        failed_cases = [tc for tc in self.results['test_cases'] if not tc['correct']]
        
        if not failed_cases:
            print("\nAll test cases passed!")
            return
        
        print(f"\nFailed Test Cases ({len(failed_cases)} cases):")
        print("="*70)
        
        for idx, tc in enumerate(failed_cases, 1):
            print(f"\n{idx}. {tc['sentence']}")
            print(f"   Expected: {'VALID' if tc['expected'] else 'INVALID'}")
            print(f"   Actual:   {'VALID' if tc['actual'] else 'INVALID'}")
            print(f"   Category: {tc['category']}")
            if tc['error']:
                print(f"   Error: {tc['error']}")
    
    def save_report(self, filename="evaluation_report.json"):
        report = {
            'timestamp': datetime.now().isoformat(),
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
            'category_stats': self.category_stats,
            'test_cases': [
                {
                    'sentence': tc['sentence'],
                    'expected': tc['expected'],
                    'actual': tc['actual'],
                    'correct': tc['correct'],
                    'parse_time': tc['parse_time'],
                    'category': tc['category'],
                    'pattern': tc.get('pattern'),
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
    print("SEKEN App - Evaluation")
    print("Sistem Parsing Kalimat Bahasa Bali Berpredikat Frasa Presisi dengan algoritma CYK")
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
            category=tc['category']
        )
        
        status = "CORRECT = " if result['correct'] else "INCORRECT = "
        expected_str = "VALID" if result['expected'] else "INVALID"
        actual_str = "VALID" if result['actual'] else "INVALID"
        print(f"   {status} Expected: {expected_str}, Actual: {actual_str}")
        
        if result['pattern']:
            print(f"   Pattern: {result['pattern']}")
    
    evaluator.calculate_final_metrics()
    evaluator.print_summary()
    evaluator.print_failed_cases()
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