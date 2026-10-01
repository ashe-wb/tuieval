"""End-to-end and unit tests. Run from the repo root:  python -m unittest discover -s tests

Uses a temporary workspace and tests/mock_server.py on a free local port; no model or GPU needed.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
GRADERS = ("answer", "rag", "reply", "tool_call", "code")


def tuieval(ws, *args, check=True):
    env = dict(os.environ, TUIEVAL_HOME=ws, NO_COLOR="1")
    p = subprocess.run([sys.executable, "-m", "tuieval", *args], capture_output=True, text=True, env=env,
                       timeout=300)
    if check and p.returncode:
        raise AssertionError(f"tuieval {' '.join(args)} exited {p.returncode}:\n{p.stdout}\n{p.stderr}")
    return p


def read(path):
    with open(path) as f:
        return f.read()


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Mock:
    def __init__(self, packs, mode):
        self.port = free_port()
        self.proc = subprocess.Popen([sys.executable, os.path.join(HERE, "mock_server.py"), "--port", str(self.port),
                                      "--packs", packs, "--mode", mode], stdout=subprocess.DEVNULL)
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=1)
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("mock server didn't start")

    def stop(self):
        self.proc.terminate()
        self.proc.wait(10)


class EmptyWorkspace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = os.path.join(self.tmp.name, "ws")

    def tearDown(self):
        self.tmp.cleanup()

    def test_not_a_workspace(self):
        p = tuieval(self.ws, "list", check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("tuieval init", p.stderr)

    def test_zero_packs(self):
        tuieval(self.tmp.name, "init", self.ws)
        for f in ("models.toml", "packs", "graders", ".gitignore"):
            self.assertTrue(os.path.exists(os.path.join(self.ws, f)), f)
        self.assertIn("none yet", tuieval(self.ws, "list").stdout)
        self.assertIn("No packs yet", tuieval(self.ws, "selftest").stdout)
        p = tuieval(self.ws, "run", check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("new-pack", p.stderr)
        self.assertIn("no results", tuieval(self.ws, "verdict", check=False).stderr)

    def test_init_keeps_existing_files(self):
        tuieval(self.tmp.name, "init", self.ws)
        with open(os.path.join(self.ws, "models.toml"), "a") as f:
            f.write("\n# mine\n")
        self.assertIn("nothing changed", tuieval(self.ws, "init").stdout)
        self.assertIn("# mine", read(os.path.join(self.ws, "models.toml")))


class Starters(unittest.TestCase):
    """Every starter template passes selftest, so a new pack starts from a working example."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ws = os.path.join(cls.tmp.name, "ws")
        tuieval(cls.tmp.name, "init", cls.ws)
        for g in GRADERS:
            tuieval(cls.ws, "new-pack", f"demo-{g.replace('_', '-')}", "--grader", g)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_selftest_passes(self):
        out = tuieval(self.ws, "selftest").stdout
        self.assertEqual(out.count(" ok"), len(GRADERS), out)

    def test_new_pack_refuses_existing(self):
        p = tuieval(self.ws, "new-pack", "demo-answer", check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("already exists", p.stderr)

    def test_run_and_verdicts(self):
        packs = os.path.join(self.ws, "packs")
        good, bad = Mock(packs, "oracle"), Mock(packs, "wrong")
        try:
            with open(os.path.join(self.ws, "models.toml"), "a") as f:
                f.write(f'\n[servers.good]\nurl = "http://127.0.0.1:{good.port}"\n'
                        f'\n[servers.bad]\nurl = "http://127.0.0.1:{bad.port}"\n')
            tuieval(self.ws, "add", "mock", "--server", "good", "--label", "good-mock")
            tuieval(self.ws, "add", "mock", "--server", "bad", "--label", "bad-mock")
            tuieval(self.ws, "run", "--tier", "certify")
        finally:
            good.stop()
            bad.stop()
        out = tuieval(self.ws, "verdict").stdout
        good_part, bad_part = out.split("bad-mock")[0], out.split("bad-mock")[1]
        self.assertGreaterEqual(good_part.count("PASS"), len(GRADERS), out)
        self.assertNotIn("FAIL", good_part, out)
        self.assertNotIn("PASS ", bad_part, out)
        res = json.loads(read(os.path.join(self.ws, "results", "good-mock", "demo-answer.json")))
        self.assertEqual(len(res["results"]), 18)
        self.assertTrue(all(r["sent"] in ("system+user", "user") for r in res["results"]))
        tuieval(self.ws, "report", "-o", os.path.join(self.tmp.name, "report.md"))
        self.assertIn("good-mock", read(os.path.join(self.tmp.name, "report.md")))
        self.assertIn("separate the models", tuieval(self.ws, "compare", "--speed").stdout)


class WorkspaceGraders(unittest.TestCase):
    """A workspace's graders/ and a pack's grader.py are loaded, with their critical flag."""

    def test_custom_grader(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = os.path.join(tmp, "ws")
            tuieval(tmp, "init", ws)
            with open(os.path.join(ws, "graders", "shout.py"), "w") as f:
                f.write(textwrap.dedent('''
                    from tuieval.graders import grader, result

                    @grader("shout", critical=True, template={"expected": "TODO"})
                    def shout(answer, test, meta):
                        ok = answer.strip() == str(test["expected"]).upper()
                        return result(ok, "loud enough" if ok else "too quiet")
                '''))
            pack = os.path.join(ws, "packs", "loud")
            os.makedirs(pack)
            with open(os.path.join(pack, "pack.toml"), "w") as f:
                f.write('label = "Loud"\ngrader = "shout"\n[certify]\nrepeat = 9\n[gate]\nmin_accuracy = 0.5\n')
            with open(os.path.join(pack, "tests.yaml"), "w") as f:
                f.write("- {id: a, input: say hi, expected: hi, reference: HI, wrong: [hi], difficulty: easy}\n"
                        "- {id: b, input: say yo, expected: yo, reference: YO, wrong: [yo], difficulty: easy}\n")
            out = tuieval(ws, "selftest").stdout
            self.assertIn("ok", out)
            env = dict(os.environ, TUIEVAL_HOME=ws)
            code = ("from tuieval import packs; p = packs.load_packs()['loud']; "
                    "print(all(t['critical_trial'] for t in p.tests))")
            self.assertEqual(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                            env=env).stdout.strip(), "True")


class Units(unittest.TestCase):
    def test_tune_workload_needs_no_packs(self):
        from tuieval import tune
        work = tune.workload(None, 65536)
        self.assertEqual([n for n, _ in work][-1], "long")
        self.assertGreater(len(work[-1][1][0]["content"]), 30000)
        self.assertEqual(len(tune.decode_workload(None)), 3)

    def test_module_needs(self):
        from tuieval import packs
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "p")
            os.makedirs(d)
            write(os.path.join(d, "pack.toml"), 'needs = ["vision", "some_missing_module"]\n')
            write(os.path.join(d, "tests.yaml"), "- {input: hi, expected: 1}\n")
            p = packs.load_pack(d)
            self.assertEqual(p.modules, ["some_missing_module"])


if __name__ == "__main__":
    unittest.main()
