"""
End-to-end smoke test on SYNTHETIC data (no dataset, no GPU needed; ~2-4 minutes on a CPU):

        python tests/smoke_test.py

Copies the code into a temp folder, writes tiny synthetic IEMOCAP/MELD pickles, and checks the real command-line pipeline:
  * train.py writes the structured results JSON and evaluates the test set exactly once (not per epoch)
  * --span_max smaller than the longest dialogue is refused; the default (auto) is set from the data
  * auto_tune.py runs tuning -> final matrix (6 methods x seeds incl. L1+L2) -> calibration of EVERY checkpoint -> report,
    and final_results.json/csv contain every metric the SOP asks for.
The numbers produced here are meaningless (random data); only the plumbing is tested.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix='ecerc_smoke_')
DS = os.environ.get('SMOKE_DATASET', 'MELD')
NOCUDA = '--no-cuda' if DS == 'MELD' else '--no_cuda'


def run(cmd, cwd, check=True):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise AssertionError('command failed: {}\n{}'.format(' '.join(cmd), (p.stdout + p.stderr)[-2500:]))
    return p


def setUpModule():
    for item in ('MELD', 'IEMOCAP', 'tests', 'auto_tune.py', 'report.py'):
        src = os.path.join(ROOT, item)
        dst = os.path.join(TMP, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('*.pkl', '__pycache__', 'reference'))
        else:
            shutil.copy(src, dst)
    run([sys.executable, os.path.join('tests', 'make_synthetic_data.py'), '--root', TMP], TMP)


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


class TestTrainPy(unittest.TestCase):
    def test_results_json_and_single_test_evaluation(self):
        ck = os.path.join(TMP, 'a.pkl')
        run([sys.executable, 'train.py', '--epochs', '2', NOCUDA, '--lim1', '--lim2', '--lim4', '--save_model_path', ck], os.path.join(TMP, DS))
        r = json.load(open(os.path.join(TMP, 'a_results.json')))
        self.assertEqual(r['protocol']['test_evaluations_during_training'], 0)
        self.assertTrue(all('test_loss' not in h and 'test_wf1' not in h for h in r['history']))     # no test numbers per epoch
        for k in ('accuracy', 'weighted_f1', 'macro_f1', 'per_class', 'confusion_matrix', 'ece', 'nll', 'brier'):
            self.assertIn(k, r['test']); self.assertIn(k, r['valid'])
        for c, d in r['test']['per_class'].items():
            self.assertEqual(sorted(d), ['f1', 'precision', 'recall', 'support'])
        self.assertTrue(r['span']['init_is_full_history'])
        self.assertEqual(r['span']['span_max'], r['span']['max_dialogue_length'])      # auto span_max = longest dialogue
        self.assertTrue(r['span']['branches']['self_event']['status'].startswith('inactive'))
        self.assertEqual(len(r['span']['history']), r['epochs_run'])                    # span evolution is logged every epoch

    def test_too_small_span_max_is_refused(self):
        p = run([sys.executable, 'train.py', '--epochs', '1', NOCUDA, '--lim4', '--span_max', '3', '--save_model_path', os.path.join(TMP, 'b.pkl')],
                os.path.join(TMP, DS), check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('smaller than the longest dialogue', p.stdout + p.stderr)


class TestPipeline(unittest.TestCase):
    def test_auto_tune_final_matrix_and_report(self):
        run([sys.executable, 'auto_tune.py', DS, '--quick', '--tune_seed', '0', '--tune_seeds', '0', '1', '--seeds', '0', '1',
             '--extra', '--epochs 2 ' + NOCUDA, '--calib_args', '--mc_passes 3 --no_cuda'], TMP)
        out = os.path.join(TMP, 'tuning_results', DS)
        res = json.load(open(os.path.join(out, 'final_results.json')))
        agg = res['aggregate']
        self.assertEqual(list(agg), ['baseline', 'L1', 'L2', 'L1+L2', 'L4', 'ALL (L1+L2+L4)'])    # L1+L2 is in the final matrix
        for m, a in agg.items():
            self.assertEqual(a['n_seeds'], 2, m)
            for k in ('accuracy', 'weighted_f1', 'macro_f1', 'ece', 'nll', 'brier'):
                self.assertIn('std', a['test'][k])
            self.assertTrue(a['per_class'] and a['confusion_matrix_mean'])
            self.assertEqual(a['calibration']['Uncalibrated']['ece']['n'], 2, m)                  # EVERY seed was calibrated, every method
            self.assertEqual(a['calibration']['Temp scaling']['nll']['n'], 2, m)
            self.assertTrue(a['pairs'])                                                           # per-pair calibration with counts
        self.assertIn('span', agg['L4']); self.assertIn('span', agg['ALL (L1+L2+L4)']); self.assertNotIn('span', agg['baseline'])
        sp = agg['L4']['span']['branches']
        self.assertEqual(sorted(b for b, v in sp.items() if v['status'].startswith('active')), ['contagion', 'cross_emotion', 'cross_event'])   # 3 of 4 active
        self.assertTrue(sp['self_event']['status'].startswith('inactive'))
        # completeness status: every required result exists -> COMPLETE (and the process exited with code 0, see run())
        st = json.load(open(os.path.join(out, 'status.json')))
        self.assertTrue(st['complete'] and st['status'] == 'COMPLETE' and not st['missing'] and not st['failures'])
        self.assertEqual((st['expected_runs'], st['trained_runs'], st['expected_calibrations'], st['calibrations_produced']), (12, 12, 12, 12))
        self.assertTrue(os.path.exists(os.path.join(out, 'STATUS_COMPLETE.txt'))); self.assertFalse(os.path.exists(os.path.join(out, 'STATUS_INCOMPLETE.txt')))
        rows = list(csv.DictReader(open(os.path.join(out, 'final_results.csv'))))
        self.assertEqual(len(rows), 12)
        for col in ('test_macro_f1', 'test_brier', 'ece_Temp_scaling', 'nll_MC-Dropout_plus_TS'):
            self.assertIn(col, rows[0])
        md = open(os.path.join(out, 'final_report.md')).read()
        for heading in ('## 1. Classification', '## 2a. Per-class F1', '## 2b. Per-class precision', '## 2c. Per-class recall', '## 3. Confusion matrices',
                        '### 4a. Deterministic', '### 4b. Stochastic', '## 5. Per-pair calibration', '## 6. Learned attention spans'):
            self.assertIn(heading, md)
        # no test numbers appear in the tuning grid
        grid = open(os.path.join(out, 'grid.csv')).read().splitlines()[0]
        self.assertNotIn('test', grid)

    def test_zz_failed_calibration_is_never_reported_as_success(self):
        """simulate a calibration failure for every configuration: exit code 2, INCOMPLETE status and banner, nothing marked successful"""
        out = os.path.join(TMP, 'tuning_results', DS)
        shutil.rmtree(os.path.join(out, 'calib'))                      # so --resume cannot reuse earlier calibrations
        p = run([sys.executable, 'auto_tune.py', DS, '--final_only', '--resume', '--seeds', '0', '--calib_args', '--no_such_flag', '--calib_retries', '0']   # single-word value: exercises the argv normalisation
             , TMP, check=False)
        self.assertEqual(p.returncode, 2, (p.stdout + p.stderr)[-1500:])
        self.assertIn('RESULTS INCOMPLETE', p.stdout)
        st = json.load(open(os.path.join(out, 'status.json')))
        self.assertFalse(st['complete']); self.assertEqual(st['status'], 'INCOMPLETE')
        self.assertEqual(len(st['missing']), 6)                         # all 6 methods x 1 seed lack their calibration
        self.assertEqual(st['calibrations_produced'], 0)
        self.assertEqual(len(st['failures']), 6)
        self.assertTrue(os.path.exists(os.path.join(out, 'STATUS_INCOMPLETE.txt'))); self.assertFalse(os.path.exists(os.path.join(out, 'STATUS_COMPLETE.txt')))
        md = open(os.path.join(out, 'final_report.md')).read()
        self.assertIn('RESULTS INCOMPLETE', md); self.assertIn('NOT AVAILABLE', md)
        self.assertTrue(json.load(open(os.path.join(out, 'final_results.json')))['meta']['status']['status'] == 'INCOMPLETE')


if __name__ == '__main__':
    unittest.main(verbosity=2)
