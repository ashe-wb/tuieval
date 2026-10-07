"""End-to-end and unit tests. Run from the repo root:  python -m unittest discover -s tests

Uses a temporary workspace and tests/mock_server.py on a free local port; no model or GPU needed.
"""
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import types
import time
import unittest
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
GRADERS = ("answer", "rag", "reply", "tool_call", "code")


def tuieval(ws, *args, check=True, ports=""):
    """Run the CLI. ports: where it may look for running servers (never the machine's real ones)."""
    env = dict(os.environ, TUIEVAL_HOME=ws, NO_COLOR="1", TUIEVAL_DETECT_PORTS=str(ports))
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
    def __init__(self, packs, mode, port=None, delay=0.0):
        self.port = port or free_port()
        self.proc = subprocess.Popen([sys.executable, os.path.join(HERE, "mock_server.py"), "--port", str(self.port),
                                      "--packs", packs, "--mode", mode, "--delay", str(delay)], stdout=subprocess.DEVNULL)
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
        self.assertIn("none yet", tuieval(self.ws, "list", "--packs").stdout)
        self.assertIn("Models: none yet", tuieval(self.ws, "list").stdout)
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
        summary, out = out.split("Details", 1)
        self.assertIn("good-mock: ready for", summary)
        self.assertIn("bad-mock: not ready for", summary)
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
            with open(os.path.join(ws, "graders", "shout.py"), "a") as f:
                f.write(textwrap.dedent('''

                    from tuieval.graders import scorecard_column

                    @scorecard_column("loud answers")
                    def loud(rows):
                        mine = [r for r in rows if r["suite"] == "loud"]
                        return f"{sum(r['ok'] for r in mine)}/{len(mine)}" if mine else None
                '''))
            out = tuieval(ws, "selftest").stdout
            self.assertIn("ok", out)
            code = ("from tuieval import packs, compare; packs.load_packs(); "
                    "rows = [{'model': 'm', 'suite': 'loud', 'test': 'loud: a', 'ok': True, 'score': 1.0, "
                    "'error': False, 'truncated': False, 'output': 'HI'}]; "
                    "h, t = compare.scorecard(rows); print(h[h.index('loud answers')], t[0][h.index('loud answers')])")
            self.assertEqual(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                            env=dict(os.environ, TUIEVAL_HOME=ws)).stdout.strip(), "loud answers 1/1")
            env = dict(os.environ, TUIEVAL_HOME=ws)
            code = ("from tuieval import packs; p = packs.load_packs()['loud']; "
                    "print(all(t['critical_trial'] for t in p.tests))")
            self.assertEqual(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                            env=env).stdout.strip(), "True")


class Units(unittest.TestCase):
    def test_settings_note_names_only_what_differs(self):
        from tuieval import compare
        infos = [{"label": m, "settings": {"temperature": 1.0, **extra}} for m, extra in
                 (("a", {}), ("b", {}), ("c", {"effort": "medium"}), ("d", {"effort": "medium"}),
                  ("e", {"effort": "medium"}), ("f", {"effort": "medium"}), ("g", {}), ("h", {}), ("i", {}),
                  ("j", {"penalty": 1.5}))]
        self.assertEqual(compare.settings_notes(infos),
                         ["Settings differ: effort medium on 4 models; penalty 1.5 on j."])

    def test_eta_goes_pack_by_pack(self):
        from tuieval.tui import RunScreen
        job = lambda key, label, total, done=0, status="waiting": types.SimpleNamespace(
            key=key, label=label, total=total, done=done, status=status)
        run = RunScreen.__new__(RunScreen)
        run.parallel, run.rough = 1, {"local/new"}
        run.jobs = [job("hosted/a", "hosted", 100, 100, "done"), job("local/a", "local", 200, 4, "running"),
                    job("local/new", "local", 50)]
        run.spr = {"local/a": 60.0, "local/new": 30.0}            # from history
        run.live = {"local/a": [900.0, 800.0, 700.0, 1000.0]}       # its first answers are the slow ones
        left, rough = run.eta_seconds()
        self.assertEqual((left, rough), (196 * 60 + 50 * 30, True))  # history, not 4 slow answers: ~3.7h, not ~47h
        run.live["local/a"].append(600.0)                            # 5 answers: its own average takes over
        run.jobs[1].done = 5
        self.assertEqual(run.eta_seconds()[0], 195 * 800 + 50 * 30)
        run.parallel = 2
        run.jobs.append(job("other/a", "other", 10))
        run.spr["other/a"] = 10.0
        self.assertEqual(run.eta_seconds()[0], (195 * 800 + 50 * 30 + 100) / 2)   # two models at a time

    def test_server_errors_never_judge_the_model(self):
        from tuieval import client
        no_endpoints = 'server returned HTTP 404: {"error":{"message":"No endpoints found for some/model."}}'
        for reason, retry, unavailable in ((no_endpoints, False, True), ("server returned HTTP 402: no credits", False, True),
                                           ("server returned HTTP 503: busy", True, False),
                                           ("server returned HTTP 400: bad request", False, False),
                                           ("connection error: timed out", False, False)):   # the model too slow
            self.assertEqual((client.is_server_error(reason), client.is_unavailable(reason)), (retry, unavailable), reason)
            self.assertEqual(client.server_error_row({"pass": False, "reason": reason}), retry or unavailable, reason)

    def test_tune_workload_needs_no_packs(self):
        from tuieval import tune
        work = tune.workload(None, 65536)
        self.assertEqual([n for n, _ in work][-1], "long")
        self.assertGreater(len(work[-1][1][0]["content"]), 30000)
        self.assertEqual(len(tune.decode_workload(None)), 3)

    def test_server_error_prefers_the_specific_line(self):
        from tuieval import engine
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "server.log")
            write(log, "loading\nerror: this GGUF stores tensors it cannot load; choose another variant\n"
                       "error: model failed to load\n")
            self.assertEqual(engine.server_error(log, 0), "this GGUF stores tensors it cannot load; choose another variant")
            write(log, "ok\nerror: runtime bootstrap failed [x]: context does not fit\n")
            self.assertEqual(engine.server_error(log, 0), "context does not fit")

    def test_test_note_skips_what_the_id_says(self):
        from tuieval.tui import test_note
        self.assertEqual(test_note({"id": "normal-entry-1-v2", "category": "normal entry",
                                    "description": "normal entry #1, compact JSON, shuffled keys"}), "compact JSON, shuffled keys")
        self.assertEqual(test_note({"id": "stale-quote", "description": "stale quote", "category": "data"}), "data")
        self.assertEqual(test_note({"id": "todo", "description": "To-do list app", "category": ""}), "To-do list app")

    def test_help_text_renders(self):
        from rich.text import Text
        from tuieval.tui import GLOSSARY, HELP
        for name, text in {**HELP, "glossary": GLOSSARY}.items():
            plain = Text.from_markup(text).plain    # raises on broken markup
            self.assertNotIn("()", plain, name)     # e.g. a [gate] swallowed as a style tag
        self.assertIn("[gate]", Text.from_markup(GLOSSARY).plain)

    def test_plain_summary(self):
        from tuieval.verdict import Verdict, plain_summary
        ok = Verdict("PASS", ["fine"], {"certified": True})
        crit = Verdict("FAIL", ["2 critical failures (e.g. x: y)"], {"critical_failures": 2})
        screened = Verdict("INCONCLUSIVE", ["screened only (3/10 tests); promising, run Certify to decide"],
                           {"certified": False, "tests_run": 3, "tests_total": 10, "repeats": 1, "want_repeat": 3})
        table = {"m": {"Support": (ok, {"support": ok}), "Trading": (crit, {"trading": crit}),
                       "Coding": (screened, {"coding": screened})}}
        (label, sentence, todo), = plain_summary(table)
        self.assertEqual(sentence, "ready for Support; not ready for Trading (2 critical failures); "
                                   "not decided yet for Coding")
        self.assertEqual(todo, [("Coding", "run Certify (the sample screened so far looks promising)", ["coding"])])

    def test_code_checks_cannot_rebind_candidate_names(self):
        from tuieval.graders.code import grade_code
        answer = "```python\nfrom datetime import datetime\ndef f(s):\n    return datetime.fromisoformat(s)\n```"
        tests = ("# SETUP\nimport datetime\n# CHECK: parse\n"
                 "assert f('2026-01-02') == datetime.datetime(2026, 1, 2)\n")
        out = grade_code(answer, {"hidden_tests": tests}, {})
        self.assertTrue(out["pass"], out)

    def test_short_note(self):
        from tuieval import tui
        self.assertEqual(tui.short_note("server exited with code 1 while loading: bad file"), "didn't start: bad file")
        self.assertTrue(tui.short_note("word " * 30).endswith("…"))

    def test_module_needs(self):
        from tuieval import packs
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "p")
            os.makedirs(d)
            write(os.path.join(d, "pack.toml"), 'needs = ["vision", "some_missing_module"]\n')
            write(os.path.join(d, "tests.yaml"), "- {input: hi, expected: 1}\n")
            p = packs.load_pack(d)
            self.assertEqual(p.modules, ["some_missing_module"])


class PickTests(unittest.TestCase):
    """Running only some tests of a pack: partial runs add up, --force redoes just the picked ones."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = os.path.join(self.tmp.name, "ws")
        tuieval(self.tmp.name, "init", self.ws)
        for name in ("apps", "other"):
            d = os.path.join(self.ws, "packs", name)
            os.makedirs(d)
            write(os.path.join(d, "pack.toml"), f'label = "{name}"\nscreen = 1\n[certify]\nrepeat = 1\n'
                                                '[gate]\nmin_accuracy = 0.1\n')
            write(os.path.join(d, "tests.yaml"), "".join(
                f"- {{id: {t}, input: {name} question {t}, expected: 1, reference: 'ANSWER: 1', wrong: ['ANSWER: 2'], "
                "difficulty: easy}\n" for t in ("a", "b", "c", "d")))
        self.mock, self.port = None, free_port()
        with open(os.path.join(self.ws, "models.toml"), "a") as f:
            f.write(f'\n[servers.m]\nurl = "http://127.0.0.1:{self.port}"\n')

    def tearDown(self):
        if self.mock:
            self.mock.stop()
        self.tmp.cleanup()

    def server(self, mode):
        """(Re)start the mock on the same port: oracle answers right, wrong answers wrong."""
        first = self.mock is None
        if self.mock:
            self.mock.stop()
        self.mock = Mock(os.path.join(self.ws, "packs"), mode, port=self.port)
        if first:
            tuieval(self.ws, "add", "mock", "--server", "m", "--label", "m")

    def rows(self, pack="apps"):
        path = os.path.join(self.ws, "results", "m", f"{pack}.json")
        if not os.path.isfile(path):
            return {}
        return {r["test"]: r["pass"] for r in json.loads(read(path))["results"]}

    def status(self, pack="apps"):
        code = f"from tuieval import engine; print(engine.Engine().result_status('m', '{pack}'))"
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              env=dict(os.environ, TUIEVAL_HOME=self.ws)).stdout.strip()

    def test_pick(self):
        from tuieval import packs
        p = packs.load_pack(os.path.join(self.ws, "packs", "apps"))
        self.assertEqual([t["id"] for t in p.pick(["c", "a"])], ["a", "c"])   # the pack's order
        with self.assertRaises(packs.PackError) as cm:
            p.pick(["a", "zz"])
        self.assertIn("'zz'", str(cm.exception))
        self.assertIn("a, b, c, d", str(cm.exception))

    def test_partial_runs_add_up_and_force_redoes_only_the_picked(self):
        self.server("oracle")
        tuieval(self.ws, "run", "--packs", "apps", "--tests", "b")
        self.assertEqual(self.rows(), {"b": True})
        self.assertEqual(self.rows("other"), {})
        tuieval(self.ws, "run", "--tests", "apps:a,apps:c", "--tier", "certify")   # --packs from --tests
        self.assertEqual(self.rows(), {"a": True, "b": True, "c": True})
        self.assertEqual(self.status(), "screened")
        out = tuieval(self.ws, "run", "--tests", "apps:b").stdout
        self.assertIn("done earlier", out)                                       # nothing missing: not re-asked
        tuieval(self.ws, "run", "--tests", "apps:d", "--tier", "certify")
        self.assertEqual(self.status(), "certified")

        self.server("wrong")
        tuieval(self.ws, "run", "--tests", "apps:b", "--force")
        self.assertEqual(self.rows(), {"a": True, "b": False, "c": True, "d": True})
        hist = os.listdir(os.path.join(self.ws, "results", "m", "history"))
        self.assertTrue(any(f.startswith("apps-") for f in hist), hist)
        self.assertEqual(self.status(), "certified")

    def test_cli_errors(self):
        self.server("oracle")
        for args, msg in ((["--tests", "a"], "say which pack"),
                          (["--packs", "apps,other", "--tests", "a"], "say which pack"),
                          (["--tests", "nope:a"], "unknown pack 'nope'"),
                          (["--packs", "apps", "--tests", "zz"], "has no test 'zz'"),
                          (["--packs", "other", "--tests", "apps:a"], "aren't in --packs")):
            p = tuieval(self.ws, "run", *args, check=False)
            self.assertNotEqual(p.returncode, 0, args)
            self.assertIn(msg, p.stderr, args)
        self.assertEqual(self.rows(), {})

    def test_presets_keep_picked_tests(self):
        from tuieval import engine
        path = os.path.join(self.ws, "presets.toml")
        engine.save_preset(path, "quick", ["m"], ["apps", "other"], None, {"apps": ["b", "d"], "gone": ["x"]})
        presets = engine.load_presets(path)
        self.assertEqual(presets["quick"]["tests"], {"apps": ["b", "d"]})
        cfg = {"models": [{"label": "m", "tags": []}]}
        self.assertEqual(engine.resolve_preset(cfg, presets["quick"], ["apps", "other"]),
                         (["m"], ["apps", "other"], None, {"apps": ["b", "d"]}))
        self.server("oracle")
        tuieval(self.ws, "run", "--preset", "quick")
        self.assertEqual(self.rows(), {"b": True, "d": True})
        self.assertEqual(len(self.rows("other")), 1)     # no pick: the Screen sample (screen = 1)


class Parallel(unittest.TestCase):
    """Several models at a time (parallel_models / --parallel): each runs in its own lane on its own
    port, answers note the models they ran alongside, and the scheduler keeps the memory and
    before_start rules."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = os.path.join(self.tmp.name, "ws")
        tuieval(self.tmp.name, "init", self.ws)
        d = os.path.join(self.ws, "packs", "apps")
        os.makedirs(d)
        write(os.path.join(d, "pack.toml"), 'label = "apps"\n[certify]\nrepeat = 1\n[gate]\nmin_accuracy = 0.1\n')
        write(os.path.join(d, "tests.yaml"), "".join(
            f"- {{id: {t}, input: question {t}, expected: 1, reference: 'ANSWER: 1', difficulty: easy}}\n"
            for t in "abcdef"))
        mock = [sys.executable, os.path.join(HERE, "mock_server.py"), "--port", "{port}", "--packs",
                os.path.join(self.ws, "packs"), "--model", "{served_name}", "--delay", "0.15"]
        with open(os.path.join(self.ws, "models.toml"), "a") as f:
            f.write(f"\n[servers.mk]\ncmd = {json.dumps(mock)}\nport = {free_port()}\n"
                    f"\n[servers.solo]\ncmd = {json.dumps(mock)}\nport = {free_port()}\n"
                    f'before_start = ["{sys.executable}", "-c", "pass"]\n')
            for label, server in (("a", "mk"), ("b", "mk"), ("s", "solo")):
                f.write(f'\n[[models]]\nlabel = "{label}"\nserver = "{server}"\nmodel = "{label}"\n')

    def tearDown(self):
        self.tmp.cleanup()

    def results(self, label):
        return json.loads(read(os.path.join(self.ws, "results", label, "apps.json")))

    def engine(self):
        from tuieval import engine
        events = []
        e = engine.Engine(lambda kind, **d: events.append((kind, d)), root=self.ws)
        return e, events

    def test_default_is_one_at_a_time(self):
        out = tuieval(self.ws, "run", "--only", "a,b", "--tier", "certify").stdout
        self.assertNotIn("at a time", out)
        for label in "ab":
            data = self.results(label)
            self.assertEqual(sum(r["pass"] for r in data["results"]), 6)
            self.assertFalse(any("ran_alongside" in r for r in data["results"]))
            self.assertNotIn("loaded_alongside", data["run"])

    def test_regrade_one_model(self):
        tuieval(self.ws, "run", "--only", "a,b", "--tier", "certify")
        tests = os.path.join(self.ws, "packs", "apps", "tests.yaml")
        write(tests, read(tests).replace("expected: 1", "expected: 2"))   # a grader change: every answer now wrong
        out = tuieval(self.ws, "regrade", "--only", "a", "--packs", "apps").stdout
        self.assertIn("6 -> 0 passed", out)
        self.assertNotIn(os.path.join("b", "apps.json"), out)
        self.assertEqual(sum(r["pass"] for r in self.results("a")["results"]), 0)
        self.assertEqual(sum(r["pass"] for r in self.results("b")["results"]), 6)   # left as it was
        for args, name in ((["--only", "a,nope"], "nope"), (["--packs", "nope"], "nope")):
            p = tuieval(self.ws, "regrade", *args, check=False)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn(f"no results for {name}", p.stderr)
        self.assertEqual(sum(r["pass"] for r in self.results("b")["results"]), 6)   # a typo regrades nothing

    def test_side_by_side_on_their_own_ports(self):
        out = tuieval(self.ws, "run", "--only", "a,b", "--tier", "certify", "--parallel", "2").stdout
        self.assertIn("up to 2 models at a time", out)
        self.assertIn("apps [", out)                    # one line per answer, with the model and pack
        for label, other in (("a", "b"), ("b", "a")):
            data = self.results(label)
            self.assertEqual(sum(r["pass"] for r in data["results"]), 6)
            self.assertTrue(any(r.get("ran_alongside") == [other] for r in data["results"]), data["results"])
        logs = read(os.path.join(self.ws, "logs", "server", "a.log")) + read(os.path.join(self.ws, "logs", "server", "b.log"))
        ports = set(re.findall(r"mock server on 127.0.0.1:(\d+)", logs))
        self.assertEqual(len(ports), 2, logs)          # the second model got a free port

    def test_machine_setting_and_cli_override(self):
        e, _ = self.engine()
        self.assertEqual(e.parallel_models(), 1)
        with open(os.path.join(self.ws, "models.toml"), "a") as f:
            f.write(f"\n[machines.{e.machine().id}]\nparallel_models = 2\n")
        out = tuieval(self.ws, "run", "--only", "a,b", "--tier", "certify").stdout
        self.assertIn("up to 2 models at a time", out)
        out = tuieval(self.ws, "run", "--only", "a,b", "--tier", "certify", "--force", "--parallel", "1").stdout
        self.assertNotIn("at a time", out)
        self.assertFalse(any("ran_alongside" in r for r in self.results("a")["results"]))

    def test_before_start_runs_alone(self):
        e, events = self.engine()
        jobs = e.plan(["a", "s"], ["apps"], None, "certify", False, {})
        e.run(jobs, parallel=2)
        self.assertTrue(all(j.status == "done" for j in jobs), [(j.key, j.status, j.note) for j in jobs])
        waits = [d["message"] for k, d in events if k == "model_waiting"]
        self.assertTrue(waits and "runs on its own" in waits[0], waits)
        for label in "as":
            self.assertFalse(any("ran_alongside" in r for r in self.results(label)["results"]))

    def test_waits_for_memory(self):
        e, events = self.engine()
        e.memory_need_gb = lambda m: 0.6 * e.memory_available_gb()   # two don't fit together
        jobs = e.plan(["a", "b"], ["apps"], None, "certify", False, {})
        e.run(jobs, parallel=2)
        self.assertTrue(all(j.status == "done" for j in jobs))
        waits = [d["message"] for k, d in events if k == "model_waiting"]
        self.assertTrue(waits and "starts when there's room" in waits[0], waits)
        self.assertFalse(any("ran_alongside" in r for r in self.results("b")["results"]))

    def test_tui_runs_side_by_side(self):
        # slower answers, so the two models surely overlap on a slow CI machine
        path = os.path.join(self.ws, "models.toml")
        write(path, read(path).replace('"--delay", "0.15"', '"--delay", "1.0"'))
        code = textwrap.dedent("""
            import asyncio, json, time
            from tuieval.tui import EvalsApp, RunScreen
            from textual.widgets import Input, SelectionList
            async def go():
                app = EvalsApp({})
                async with app.run_test(size=(160, 50)) as pilot:
                    await pilot.pause()
                    s = app.screen
                    s.query_one("#suites", SelectionList).select("apps")
                    s.selected_models = {"a", "b"}
                    s.refresh_models()
                    s.query_one("#tier-certify").value = True
                    s.query_one("#parallel", Input).value = "2"
                    await pilot.pause()
                    est = str(s.query_one("#estimate").render())
                    s.action_start()
                    deadline = time.time() + 180     # CI machines can take long to start servers
                    while time.time() < deadline:
                        await pilot.pause(0.1)
                        if isinstance(app.screen, RunScreen) and len(app.screen.active()) == 2 \\
                                and all(j.status == "running" for j in app.screen.jobs):
                            break
                    run = app.screen
                    first = run.watch
                    await pilot.press("v")
                    switched = run.watch
                    deadline = time.time() + 240
                    while run.running and time.time() < deadline:
                        await pilot.pause(0.2)
                    print(json.dumps({"est": est, "first": first, "switched": switched,
                                      "notes": [(j.key, j.status, j.note) for j in run.jobs],
                                      "status": sorted({j.status for j in run.jobs}),
                                      "parallel": app.load_state().get("parallel")}))
            asyncio.run(go())
        """)
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=480,
                           env=dict(os.environ, TUIEVAL_HOME=self.ws, TUIEVAL_DETECT_PORTS=""))
        out = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else p.stderr
        self.assertIsInstance(out, dict, out)
        self.assertIn("Up to 2 models at a time", out["est"])
        self.assertEqual(out["status"], ["done"], out["notes"])
        self.assertEqual(out["parallel"], 2)
        self.assertIn(out["first"], ("a", "b"))
        self.assertEqual({out["first"], out["switched"]}, {"a", "b"})   # v streams the other model
        self.assertTrue(any(r.get("ran_alongside") for r in self.results("a")["results"]))

    def test_skip_one_model_keeps_the_other(self):
        e, events = self.engine()
        jobs = e.plan(["a", "b"], ["apps"], None, "certify", False, {})
        orig = e.on_event

        def on_event(kind, **d):
            orig(kind, **d)
            if kind == "request_done" and d["job"].label == "a" and d["job"].done == 1:
                threading.Thread(target=e.skip_model, args=("a",)).start()
        e.on_event = on_event
        e.run(jobs, parallel=2)
        status = {j.label: j.status for j in jobs}
        self.assertEqual(status, {"a": "skipped", "b": "done"}, [(j.key, j.status, j.note) for j in jobs])
        self.assertEqual(sum(r["pass"] for r in self.results("b")["results"]), 6)


class AddToRun(unittest.TestCase):
    """More evals while a run is going: added to it (they run after its other jobs) or queued after it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = os.path.join(self.tmp.name, "ws")
        tuieval(self.tmp.name, "init", self.ws)
        for name in ("apps", "other"):
            d = os.path.join(self.ws, "packs", name)
            os.makedirs(d)
            write(os.path.join(d, "pack.toml"), f'label = "{name}"\n[certify]\nrepeat = 1\n[gate]\nmin_accuracy = 0.1\n')
            write(os.path.join(d, "tests.yaml"), "".join(
                f"- {{id: {t}, input: {name} question {t}, expected: 1, reference: 'ANSWER: 1', difficulty: easy}}\n"
                for t in "abcd"))
        mock = [sys.executable, os.path.join(HERE, "mock_server.py"), "--port", "{port}", "--packs",
                os.path.join(self.ws, "packs"), "--model", "{served_name}", "--delay", "0.2"]
        with open(os.path.join(self.ws, "models.toml"), "a") as f:
            f.write(f"\n[servers.mk]\ncmd = {json.dumps(mock)}\nport = {free_port()}\n")
            for label in "ab":
                f.write(f'\n[[models]]\nlabel = "{label}"\nserver = "mk"\nmodel = "{label}"\n')

    def tearDown(self):
        self.tmp.cleanup()

    def finish(self, e, t, limit=480):
        """Wait for a run thread; CI machines can take half a minute per server start. One that's
        still going after `limit` is cancelled, so it can't run into the next test."""
        t.join(limit)
        if t.is_alive():
            e.cancel()
            t.join(120)
            self.fail(f"the run didn't finish in {limit}s")

    def test_engine_adds_jobs_to_the_run_going(self):
        from tuieval import engine
        events = []
        e = engine.Engine(lambda kind, **d: events.append((kind, d)), root=self.ws)
        jobs = e.plan(["a"], ["apps"], None, "certify", False, {})
        self.assertEqual(e.add_jobs(e.plan(["b"], ["apps"], None, "certify", False, {})),
                         ([], "the run is finishing"))                     # no run yet
        t = threading.Thread(target=e.run, args=(jobs,))
        t.start()
        deadline = time.time() + 180   # CI machines can take long to start servers
        while not any(k == "request_done" for k, _ in events) and time.time() < deadline:
            time.sleep(0.05)
        added, why = e.add_jobs(e.plan(["b", "a"], ["apps", "other"], None, "certify", False, {}))
        self.assertIsNone(why)
        self.assertEqual(sorted(j.key for j in added), ["a/other", "b/apps", "b/other"])   # a/apps is running
        self.assertEqual(e.add_jobs(e.plan(["b"], ["apps"], None, "certify", False, {}))[1],
                         "everything picked is already in the run")
        self.finish(e, t)
        self.assertEqual([j.status for j in jobs], ["done"] * 4)
        self.assertEqual(jobs[0].key, "a/apps")
        started = [d["job"].key for k, d in events if k == "job_started"]
        self.assertEqual(started[0], "a/apps")
        self.assertEqual(e.add_jobs(e.plan(["b"], ["apps"], None, "certify", True, {}))[1], "the run is finishing")
        queue_done = [d for k, d in events if k == "queue_done"]
        self.assertEqual(len(queue_done[0]["jobs"]), 4)

    def test_added_models_use_a_free_lane(self):
        from tuieval import engine
        path = os.path.join(self.ws, "models.toml")    # slower answers: a is surely still answering when b is added
        write(path, read(path).replace('"--delay", "0.2"', '"--delay", "1.0"'))
        events = []
        e = engine.Engine(lambda kind, **d: events.append((kind, d)), root=self.ws)
        jobs = e.plan(["a"], ["apps"], None, "certify", False, {})
        t = threading.Thread(target=e.run, args=(jobs, 2))
        t.start()
        deadline = time.time() + 180   # CI machines can take long to start servers
        while not any(k == "request_done" for k, _ in events) and time.time() < deadline:
            time.sleep(0.05)
        added, why = e.add_jobs(e.plan(["a", "b"], ["other", "apps"], None, "certify", False, {}))
        self.assertEqual(sorted(j.key for j in added), ["a/other", "b/apps", "b/other"])
        self.finish(e, t)
        self.assertEqual([j.status for j in jobs], ["done"] * 4)
        order = [(k, d.get("label") or d["job"].key) for k, d in events if k in ("model_loading", "job_done")]
        self.assertLess(order.index(("model_loading", "b")), order.index(("job_done", "a/apps")), order)  # next to a
        a_other = json.loads(read(os.path.join(self.ws, "results", "a", "other.json")))["results"]
        self.assertFalse(any("a" in (r.get("ran_alongside") or []) for r in a_other))
        loads = [d["label"] for k, d in events if k == "model_loading"]
        self.assertEqual(loads.count("a"), 2)                                 # a's new pack: after it ended

    def test_tui_adds_or_queues_while_running(self):
        code = textwrap.dedent("""
            import asyncio, json, time
            from tuieval.tui import EvalsApp, RunScreen, SetupScreen, ChoiceScreen
            from textual.widgets import SelectionList
            async def go():
                app = EvalsApp({})
                async with app.run_test(size=(160, 50)) as pilot:
                    await pilot.pause()
                    s = app.screen
                    s.query_one("#suites", SelectionList).select("apps")
                    s.selected_models = {"a"}
                    s.refresh_models()
                    s.query_one("#tier-certify").value = True
                    await pilot.pause()
                    s.action_start()
                    deadline = time.time() + 180     # CI machines can take long to start servers
                    while time.time() < deadline:
                        await pilot.pause(0.1)
                        if isinstance(app.screen, RunScreen) and app.screen.counts()[0]:
                            break
                    run = app.screen
                    await pilot.press("n")                       # more evals while it runs
                    await pilot.pause()
                    on_setup = isinstance(app.screen, SetupScreen)
                    app.screen.selected_models = {"b"}
                    app.screen.refresh_models()
                    app.screen.action_start()
                    await pilot.pause()
                    asked = isinstance(app.screen, ChoiceScreen)
                    app.screen.dismiss("add")
                    await pilot.pause()
                    app.screen.query_one("#suites", SelectionList).deselect("apps")
                    app.screen.query_one("#suites", SelectionList).select("other")
                    app.screen.selected_models = {"a"}
                    app.screen.refresh_models()
                    app.screen.action_start()
                    await pilot.pause()
                    app.screen.dismiss("queue")
                    await pilot.pause()
                    app.screen.query_one("#tier-smoke").value = True    # another tier can't join: queued
                    await pilot.pause()
                    app.screen.action_start()
                    await pilot.pause()
                    other_tier = not isinstance(app.screen, ChoiceScreen)
                    queued = len(app.queue)
                    deadline = time.time() + 300
                    while time.time() < deadline:
                        await pilot.pause(0.2)
                        if not run.running and not app.queue and not app.active_session:
                            break
                    second = app.sessions[1]
                    print(json.dumps({"on_setup": on_setup, "asked": asked, "queued": queued, "other_tier": other_tier,
                                      "run": [(j.key, j.status) for j in run.jobs], "total": run.total,
                                      "sessions": len(app.sessions),
                                      "second": [(j.key, j.status) for j in second.jobs]}))
            asyncio.run(go())
        """)
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                           env=dict(os.environ, TUIEVAL_HOME=self.ws, TUIEVAL_DETECT_PORTS=""))
        out = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else p.stderr
        self.assertIsInstance(out, dict, out)
        self.assertTrue(out["on_setup"])
        self.assertTrue(out["asked"])
        self.assertTrue(out["other_tier"])
        self.assertEqual(out["queued"], 2)
        self.assertEqual(out["run"], [["a/apps", "done"], ["b/apps", "done"]])   # added to the run going
        self.assertEqual(out["total"], 8)
        self.assertEqual(out["sessions"], 3)
        self.assertEqual(out["second"], [["a/other", "done"]])                  # queued: its own run after


class PTAIndex(unittest.TestCase):
    """The PTA index: parsimony from tokens and speed from time, each relative to the best on a log
    scale, and accuracy, all over the questions every compared model answered."""

    def test_index_compares_shared_questions(self):
        from tuieval import pta

        def row(model, test, ok, secs, repeat=0):
            return {"model": model, "suite": "p", "test": f"p: {test}", "ok": ok, "latency": secs * 1000,
                    "tokens": 10 if model == "slow" else 40, "truncated": False, "gen_tps": None, "repeat": repeat}
        rows = [row("fast", "q1", True, 2), row("fast", "q2", False, 2),
                row("slow", "q1", True, 8), row("slow", "q2", True, 4), row("slow", "q2", True, 6, repeat=1),
                row("slow", "q3", True, 100)]               # only slow answered q3: not compared
        res = pta.index(rows)
        self.assertEqual(res["questions"], 2)
        by = {x["model"]: x for x in res["models"]}
        self.assertEqual(by["fast"]["total_s"], 4)
        self.assertEqual(by["slow"]["total_s"], 13)      # q1 8 + q2 median of 4 and 6
        self.assertEqual(by["fast"]["T"], 100)
        self.assertAlmostEqual(by["slow"]["T"], 100 - 15 * math.log2(13 / 4))   # 15 points per doubling
        self.assertEqual((by["slow"]["tokens"], by["fast"]["tokens"]), (20, 80))   # medians over repeats
        self.assertEqual((by["slow"]["P"], by["fast"]["P"]), (100, 70))          # 4x the tokens: two doublings
        self.assertEqual((pta.score(2, 1), pta.score(1000, 1)), (85, 0))
        header, table = pta.table(res)
        self.assertEqual(header[1:4], ["tokens/answer", "time/answer", "answers right"])
        cells = {r[0]: r for r in table}
        self.assertEqual((cells["slow"][1], cells["fast"][1:3]), ("10 ★", ["40", "2.0s ★"]))   # per answer
        bars = pta.bars(res, 10, markup=False)
        self.assertEqual(bars[0].split(), ["P", "parsimony", "(tokens)", "T", "time", "A", "accuracy"])
        self.assertEqual((bars[0].index("P parsimony"), bars[0].index("T time")),   # each bar starts under
                         (bars[2].index("███████░░░"), bars[2].index("██████████")))  # its heading
        # the bar is the score (4x the tokens: 70); beside it, how many times the best
        self.assertEqual(bars[2].split(), ["fast", "███████░░░", "4.0×", "██████████", "1.0×", "█████░░░░░", "50.0%"])
        self.assertEqual((pta.times(1), pta.times(9.84), pta.times(27.9)), ("1.0×", "9.8×", "28×"))
        self.assertEqual((by["fast"]["A"], by["slow"]["A"]), (50, 100))
        self.assertEqual([x["model"] for x in res["models"]], ["slow", "fast"])   # best accuracy first

    def test_accuracy_never_rounds_up_to_perfect(self):
        from tuieval import compare, pta
        self.assertEqual([compare.pct(f) for f in (467 / 469, 1, 0, 0.5)], ["99.6%", "100%", "0%", "50.0%"])
        row = lambda model, a: {"model": model, "P": 100, "T": 100, "A": a, "tokens_x": 1, "time_x": 1}
        bars = pta.bars({"models": [row("two-wrong", 100 * 467 / 469), row("perfect", 100.0)]}, 10, markup=False)
        two_wrong, perfect = bars[1].split(), bars[2].split()
        self.assertEqual((two_wrong[-2], two_wrong[-1]), ("█████████░", "99.6%"))   # not a full bar
        self.assertEqual((perfect[-2], perfect[-1]), ("██████████", "100%"))

    def test_server_errors_and_models_with_few_answers(self):
        from tuieval import pta

        def row(model, test, ok=True, error=False):
            return {"model": model, "suite": "p", "test": f"p: {test}", "ok": ok, "error": error,
                    "latency": 1000, "tokens": 10, "truncated": False, "gen_tps": None, "repeat": 0}
        rows = ([row(m, f"q{i}") for m in ("a", "b") for i in range(10)] + [row("few", "q0", ok=False)]
                + [row("down", f"q{i}", ok=False, error=True) for i in range(10)]
                + [row("b", "q10", ok=False, error=True)])      # an error among real answers: not counted
        res = pta.index(rows)
        self.assertEqual([x["model"] for x in res["models"]], ["a", "b"])
        self.assertEqual(res["questions"], 10)                  # not shrunk to the one "few" answered
        self.assertEqual(res["left_out"], [("few", 1)])
        self.assertEqual(res["no_answers"], ["down"])
        self.assertEqual([x["answers"] for x in res["models"]], [10, 10])
        legend = "\n".join(pta.left_out_lines(res))
        self.assertIn("few (1 of 10)", legend)
        self.assertIn("only server errors: down", legend)
        picked = pta.index(rows, ["a", "few"])      # picked by hand: compared anyway
        self.assertEqual((picked["questions"], picked["left_out"]), (1, []))

    def test_cli_report_and_tui(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = os.path.join(tmp, "ws")
            tuieval(tmp, "init", ws)
            d = os.path.join(ws, "packs", "apps")
            os.makedirs(d)
            write(os.path.join(d, "pack.toml"), 'label = "apps"\n[certify]\nrepeat = 1\n[gate]\nmin_accuracy = 0.1\n')
            write(os.path.join(d, "tests.yaml"), "".join(
                f"- {{id: {t}, input: question {t}, expected: 1, reference: 'ANSWER: 1', wrong: ['ANSWER: 2'], "
                "difficulty: easy}\n" for t in "abcd"))
            right = Mock(os.path.join(ws, "packs"), "oracle")
            wrong = Mock(os.path.join(ws, "packs"), "wrong", delay=0.3)
            try:
                with open(os.path.join(ws, "models.toml"), "a") as f:
                    f.write(f'\n[servers.here]\nurl = "http://127.0.0.1:{right.port}"\n'
                            f'\n[servers.there]\nurl = "http://127.0.0.1:{wrong.port}"\n')
                tuieval(ws, "add", "mock", "--server", "here", "--label", "good")
                tuieval(ws, "add", "mock", "--server", "there", "--label", "hosted")
                tuieval(ws, "run", "--tier", "certify")
            finally:
                right.stop()
                wrong.stop()
            out = tuieval(ws, "pta").stdout
            self.assertIn("Compared on the 4 questions all 2 model(s) answered", out)
            bars = {l.split()[0]: l for l in out.splitlines() if "█" in l}
            self.assertEqual((bars["good"].count("1.0×"), "100%" in bars["good"]), (2, True))
            self.assertEqual(bars["hosted"].count("1.0×"), 1)   # as many tokens (the mock), but slower
            self.assertIn(" 0%", bars["hosted"])
            self.assertIn("P parsimony", out)
            self.assertIn("4/4", tuieval(ws, "pta", "--only", "good").stdout)
            report = os.path.join(tmp, "r.md")
            tuieval(ws, "report", "-o", report)
            self.assertIn("## PTA index", read(report))
            code = textwrap.dedent("""
                import asyncio, json
                from tuieval.tui import EvalsApp, ResultsScreen
                from textual.widgets import DataTable, Static, TabbedContent
                async def go():
                    app = EvalsApp({})
                    async with app.run_test(size=(180, 60)) as pilot:
                        await pilot.pause()
                        app.push_screen(ResultsScreen())
                        await pilot.pause(0.5)
                        t = app.screen.query_one("#pta", DataTable)
                        rows = [[str(c) for c in t.get_row_at(i)] for i in range(t.row_count)]
                        tri = str(app.screen.query_one("#pta-bars", Static).render())
                        app.screen.query_one("#pq-disagree").value = True       # right vs wrong on all 4
                        await pilot.pause(0.3)
                        pq = app.screen.query_one("#per-question", DataTable)
                        both = pq.row_count
                        app.pq_models = {"good"}                                # one model never disagrees
                        app.screen.fill_per_question()
                        await pilot.pause(0.3)
                        alone = pq.row_count
                        print(json.dumps({"rows": rows, "bars": "P parsimony" in tri, "disagree": [both, alone],
                                          "tab": app.screen.query_one(TabbedContent).active}))
                asyncio.run(go())
            """)
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                               env=dict(os.environ, TUIEVAL_HOME=ws, TUIEVAL_DETECT_PORTS=""))
            got = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else p.stderr
            self.assertIsInstance(got, dict, got)
            self.assertTrue(got["bars"])
            self.assertEqual(got["disagree"], [4, 0])
            self.assertEqual(got["tab"], "tab-pta")                  # Results opens on the PTA index
            self.assertEqual([(r[0], r[3]) for r in got["rows"]][0], ("good", "4/4"))


class Unavailable(unittest.TestCase):
    """A model no server will serve (OpenRouter's 404 "No endpoints found", a bad key, no credits) stops at
    once and is never judged on it; the other models run as usual."""

    def test_unavailable_model_stops_and_isnt_judged(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = os.path.join(tmp, "ws")
            tuieval(tmp, "init", ws)
            d = os.path.join(ws, "packs", "apps")
            os.makedirs(d)
            write(os.path.join(d, "pack.toml"), 'label = "apps"\n[certify]\nrepeat = 1\n[gate]\nmin_accuracy = 0.1\n')
            write(os.path.join(d, "tests.yaml"), "".join(
                f"- {{id: {t}, input: question {t}, expected: 1, reference: 'ANSWER: 1', difficulty: easy}}\n"
                for t in "abc"))
            up, down = Mock(os.path.join(ws, "packs"), "oracle"), Mock(os.path.join(ws, "packs"), "unavailable")
            try:
                with open(os.path.join(ws, "models.toml"), "a") as f:
                    f.write(f'\n[servers.up]\nurl = "http://127.0.0.1:{up.port}"\n'
                            f'\n[servers.down]\nurl = "http://127.0.0.1:{down.port}"\n')
                tuieval(ws, "add", "mock", "--server", "up", "--label", "works")
                tuieval(ws, "add", "mock", "--server", "down", "--label", "gone")
                t = time.time()
                out = tuieval(ws, "run", "--tier", "certify", check=False).stdout
            finally:
                up.stop()
                down.stop()
            self.assertLess(time.time() - t, 60)                      # no retrying for minutes
            self.assertIn("isn't available there", out)
            verdict = tuieval(ws, "verdict", check=False).stdout
            self.assertNotIn("0% right", verdict)                       # never judged on the 404s
            rows = [l for l in verdict.splitlines() if "gone" in l]
            self.assertFalse(any("FAIL" in l for l in rows), rows)
            self.assertIn("works", tuieval(ws, "pta").stdout)


class DiscreteGPU(unittest.TestCase):
    """Nvidia and AMD cards on Linux, read from their vendors' tools (faked here, so no GPU is needed)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bin = os.path.join(self.tmp.name, "bin")
        os.makedirs(self.bin)
        self.path = os.environ["PATH"]
        os.environ["PATH"] = self.bin + os.pathsep + "/usr/bin:/bin"   # only the fakes, never a real GPU tool

    def tearDown(self):
        os.environ["PATH"] = self.path
        self.tmp.cleanup()

    def tool(self, name, script):
        path = os.path.join(self.bin, name)
        write(path, "#!/bin/sh\n" + script)
        os.chmod(path, 0o755)

    def test_nvidia(self):
        from tuieval import machines
        self.tool("nvidia-smi", textwrap.dedent("""
            case "$*" in
              *query-gpu=name,memory.total*) printf 'NVIDIA GeForce RTX 4090, 24564\nNVIDIA GeForce RTX 4090, 24564\n' ;;
              *query-compute-apps*) printf '4242, 18000\n999, 3000\n' ;;
              *query-gpu=memory.used*) printf '18500\n200\n' ;;
            esac
        """))
        gpus = machines.discrete_gpus()
        self.assertEqual([n for n, _ in gpus], ["NVIDIA GeForce RTX 4090"] * 2)
        self.assertAlmostEqual(sum(g for _, g in gpus), 47.98, places=1)
        self.assertEqual(machines.gpu_id(gpus, 127.6), "2xrtx4090-128gb")
        self.assertAlmostEqual(machines.vram_used_gb([4242, 4243]), 18000 / 1024)   # the server's own
        self.assertAlmostEqual(machines.vram_used_gb(), 18700 / 1024)              # the whole machine

    def test_amd(self):
        from tuieval import machines
        self.tool("amd-smi", "exit 1\n")                                         # older ROCm: rocm-smi
        self.tool("rocm-smi", textwrap.dedent("""
            case "$*" in
              *showproductname*) echo '{"card0": {"VRAM Total Memory (B)": "25753026560", "Card Series": "Radeon RX 7900 XTX"}, "system": {}}' ;;
              *) echo '{"card0": {"VRAM Total Used Memory (B)": "4294967296"}}' ;;
            esac
        """))
        self.assertEqual(machines.discrete_gpus(), [("Radeon RX 7900 XTX", 25753026560 / 2**30)])
        self.assertEqual(machines.gpu_id(machines.discrete_gpus(), 64), "rx7900xtx-64gb")
        self.assertAlmostEqual(machines.vram_used_gb([1]), 4.0)                    # machine-wide on AMD
        self.tool("amd-smi", "echo '[{\"gpu\": 0, \"asic\": {\"market_name\": \"Radeon PRO W7900\"}, "
                             "\"vram\": {\"size\": {\"value\": 49152, \"unit\": \"MB\"}}}]'\n")
        self.assertEqual(machines.discrete_gpus(), [("Radeon PRO W7900", 48.0)])

    def test_missing_or_broken_tools_mean_no_gpu(self):
        from tuieval import machines
        self.assertEqual(machines.discrete_gpus(), [])
        self.assertIsNone(machines.vram_used_gb([1]))
        self.tool("nvidia-smi", "echo 'NVIDIA-SMI has failed because it could not communicate with the driver'\nexit 9\n")
        self.tool("rocm-smi", "echo 'not json'\n")
        self.assertEqual(machines.discrete_gpus(), [])

    def test_sizing_ids_and_overrides(self):
        from tuieval import engine, machines
        gpu = machines.Machine("rtx4090-128gb", "x86_64", 128.0, 24.0, 16, 0, None, "Linux",
                               gpu_name="NVIDIA GeForce RTX 4090", discrete=True)
        self.assertIn("NVIDIA GeForce RTX 4090 · 24 GB VRAM", gpu.summary)
        ws = os.path.join(self.tmp.name, "ws")
        tuieval(self.tmp.name, "init", ws)
        real = machines.detect
        machines.detect = lambda: gpu
        try:
            e = engine.Engine(root=ws)
            self.assertEqual((e.machine().id, e.headroom_gb(e.machine())), ("rtx4090-128gb", 1.5))
            self.assertEqual(e.memory_available_gb(), 22.5)                       # VRAM, not 128 GB of RAM
            os.makedirs(os.path.join(ws, "tuning", "x8664-128gb"))                # recorded before detection
            self.assertEqual(e.machine().id, "x8664-128gb")                        # keeps its results' id
            with open(os.path.join(ws, "models.toml"), "a") as f:
                f.write("\n[machines.x8664-128gb]\ngpu_memory_gb = 20\nmemory_headroom_gb = 2\n")
            e = engine.Engine(root=ws)
            self.assertEqual((e.machine().gpu_limit_gb, e.memory_available_gb()), (20.0, 18.0))
        finally:
            machines.detect = real
    def test_doctor_and_vram_peak(self):
        import contextlib, io, threading
        from tuieval import doctor, engine, machines
        self.tool("nvidia-smi", "printf 'NVIDIA GeForce RTX 4090, 24564\\n'\n")
        gpu = machines.Machine("x8664-128gb", "x86_64", 128.0, 24.0, 16, 0, None, "Linux",
                               gpu_name="NVIDIA GeForce RTX 4090", discrete=True)
        ws = os.path.join(self.tmp.name, "ws")
        tuieval(self.tmp.name, "init", ws)
        real, real_vram = machines.detect, machines.vram_used_gb
        machines.detect = lambda: gpu
        env = dict(os.environ)
        os.environ.update(TUIEVAL_HOME=ws, TUIEVAL_DETECT_PORTS="", NO_COLOR="1")
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                doctor.check()
            text = out.getvalue()
            self.assertIn("NVIDIA GeForce RTX 4090 · 24 GB VRAM", text)
            self.assertIn("sized to 24 GB of VRAM, keeping 1.5 GB free", text)
            self.assertIn("EVALS_MACHINE=rtx4090-128gb", text)                    # an id kept from before
            machines.vram_used_gb = lambda pids=None: 12.5
            e, info, stop = engine.Engine(root=ws), {}, threading.Event()
            proc = subprocess.Popen(["sleep", "5"])
            try:
                e._watch_memory(proc, "m", stop, [0.0], info)
                time.sleep(1)
            finally:
                stop.set()
                proc.kill()
                proc.wait()
            self.assertEqual(info.get("vram_peak_gb"), 12.5)
        finally:
            machines.detect, machines.vram_used_gb = real, real_vram
            os.environ.clear()
            os.environ.update(env)

    def test_fit_messages_name_vram(self):
        from tuieval import machines
        path = os.path.join(self.tmp.name, "m.gguf")
        fake_gguf(path, size=1000)
        gpu = lambda gb: machines.Machine("rtx4090-128gb", "x86_64", 128.0, gb, 16, 0, None, "Linux",
                                          gpu_name="NVIDIA GeForce RTX 4090", discrete=True)
        ok, no = machines.fit(path, gpu(24.0), headroom_gb=1.5), machines.fit(path, gpu(1.2), headroom_gb=0.1)
        self.assertTrue(ok.fits and ok.note.endswith("GB of VRAM)"), ok.note)
        self.assertTrue(not no.fits and no.note.endswith("GB of VRAM available"), no.note)
        mac = machines.Machine("m3max-64gb", "Apple M3 Max", 64.0, 48.0, 12, 4, 40, "macOS")
        self.assertNotIn("VRAM", machines.fit(path, mac).note)

    def test_vram_peak_reaches_the_pta_index(self):
        from tuieval import engine, pta
        sittings = [{"started": 1, "wall_s": 10, "vram_peak_gb": 18.2, "peak_rss_mb": 2048},
                    {"started": 2, "wall_s": 10, "vram_peak_gb": 21.5, "peak_rss_mb": 1900}]
        run = engine.Engine.sittings_summary(sittings)
        self.assertEqual(run["vram_peak_gb"], 21.5)                   # the peak over every sitting is kept
        infos = [{"label": "gpu", "vram_peak_gb": run["vram_peak_gb"], "peak_rss_mb": run["peak_rss_mb"]},
                 {"label": "mac", "peak_rss_mb": 25600}]
        self.assertEqual(pta.memory_gb(infos), {"gpu": 21.5, "mac": 25.0})   # VRAM where it was measured

    def test_mac_is_unchanged(self):
        from tuieval import engine, machines
        e = engine.Engine.__new__(engine.Engine)
        e.cfg = {"machines": {}}
        mac = machines.Machine("m3max-64gb", "Apple M3 Max", 64.0, 48.0, 12, 4, 40, "macOS")
        self.assertFalse(mac.discrete)                                             # Macs are unchanged
        self.assertEqual(e.headroom_gb(mac), 4.0)


class FirstRun(unittest.TestCase):
    """A new user's first run fails fast with the fix, never hangs or shows a bare Python error."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = os.path.join(self.tmp.name, "ws")
        tuieval(self.tmp.name, "init", self.ws)
        tuieval(self.ws, "new-pack", "first")
        self.port = free_port()
        with open(os.path.join(self.ws, "models.toml"), "a") as f:   # never the machine's real LM Studio port
            f.write(f'\n[servers.local]\nurl = "http://127.0.0.1:{free_port()}"\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_without_models_says_how_to_add_one(self):
        p = tuieval(self.ws, "run", check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("tuieval add", p.stderr)

    def test_a_server_that_isnt_running_fails_fast(self):
        tuieval(self.ws, "add", "some-model", "--server", "local")
        t = time.time()
        p = tuieval(self.ws, "run", "--tier", "smoke", check=False)
        self.assertLess(time.time() - t, 30)
        self.assertIn("nothing is answering at", p.stdout)
        self.assertIn("tuieval doctor", p.stdout)

    def test_a_missing_server_program_is_named(self):
        with open(os.path.join(self.ws, "models.toml"), "a") as f:
            f.write('\n[servers.gone]\ncmd = ["no-such-server-xyz", "-m", "{model}", "--port", "{port}"]\nport = 18599\n'
                    'model_is_path = true\n')
        write(os.path.join(self.tmp.name, "m.gguf"), "x")
        tuieval(self.ws, "add", os.path.join(self.tmp.name, "m.gguf"), "--server", "gone", "--label", "g")
        p = tuieval(self.ws, "run", "--tier", "smoke", check=False)
        self.assertIn("no-such-server-xyz isn't installed", p.stdout)
        self.assertNotIn("FileNotFoundError", p.stdout)

    def test_first_tui_screen_is_ready_to_start(self):
        tuieval(self.ws, "add", "my-model", "--server", "local")
        code = textwrap.dedent("""
            import asyncio
            from tuieval.tui import EvalsApp
            from textual.widgets import SelectionList
            async def go():
                app = EvalsApp({})
                async with app.run_test(size=(140, 40)) as pilot:
                    await pilot.pause()
                    s = app.screen
                    print(s.query_one("#suites", SelectionList).selected, sorted(s.selected_models), s.tier(),
                          s.check_action("tune", ()))
            asyncio.run(go())
        """)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=dict(os.environ, TUIEVAL_HOME=self.ws, TUIEVAL_DETECT_PORTS=""), timeout=60).stdout.strip()
        self.assertEqual(out, "['first'] ['my-model'] smoke False")   # tuning shows once something has run

    def test_results_open_on_pta_index(self):
        tuieval(self.ws, "add", "my-model", "--server", "local")
        code = textwrap.dedent("""
            import asyncio
            from tuieval.tui import EvalsApp
            from textual.widgets import TabbedContent, TabPane
            async def go():
                app = EvalsApp({})
                async with app.run_test(size=(140, 40)) as pilot:
                    await pilot.pause()
                    await pilot.press("r")
                    await pilot.pause()
                    tc = app.screen.query_one(TabbedContent)
                    tabs = " | ".join(str(tc.get_tab(p.id).label) for p in tc.query(TabPane))
                    print(type(app.screen).__name__, tc.active, "|", tabs)
            asyncio.run(go())
        """)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=dict(os.environ, TUIEVAL_HOME=self.ws, TUIEVAL_DETECT_PORTS=""), timeout=60).stdout.strip()
        self.assertEqual(out, "ResultsScreen tab-pta | PTA index | Production readiness | Per question | Failures")

    def test_stream_follows_only_at_the_bottom(self):
        code = textwrap.dedent("""
            import asyncio
            from textual.app import App
            from tuieval.tui import StreamView
            class A(App):
                def compose(self):
                    yield StreamView()
            async def go():
                app = A()
                async with app.run_test(size=(60, 12)) as pilot:
                    s = app.query_one(StreamView)
                    for i in range(100):
                        s.append(f"line {i}\\n")
                    await pilot.pause()
                    s.scroll_to(y=20, animate=False)
                    await pilot.pause()
                    for i in range(100):
                        s.append("more\\n")
                        await pilot.pause()
                    held = s.scroll_y
                    s.scroll_end(animate=False)
                    await pilot.pause()
                    s.append("last\\n")
                    await pilot.pause()
                    following = s.scroll_y == s.max_scroll_y
                    s.MAX_CHARS = len(s.text) + 100        # the next text trims the oldest
                    s.scroll_to(y=s.max_scroll_y - 30, animate=False)
                    await pilot.pause()
                    top = s.document.get_line(s.scroll_y)
                    s.append("x\\n" * 60)
                    await pilot.pause()
                    await pilot.pause()
                    print(held, following, s.text.startswith("…"), s.document.get_line(s.scroll_y) == top)
            asyncio.run(go())
        """)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60).stdout.strip()
        self.assertEqual(out, "20 True True True")   # scrolled up stays put, also through a trim; bottom follows

    def test_doctor(self):
        tuieval(self.ws, "add", "mock", "--server", "local", "--label", "wanted")
        p = tuieval(self.ws, "doctor", check=False, ports=self.port)
        self.assertEqual(p.returncode, 1)
        self.assertIn("✗ local: nothing answers at", p.stdout)
        mock = Mock(os.path.join(self.ws, "packs"), "oracle", port=self.port)   # a server on a detected port
        try:
            out = tuieval(self.ws, "doctor", check=False, ports=self.port).stdout
            self.assertIn(f"run `tuieval add` to add the models of the one running at http://127.0.0.1:{self.port}", out)
            text = read(os.path.join(self.ws, "models.toml"))
            write(os.path.join(self.ws, "models.toml"),
                  re.sub(r'(\[servers\.local\]\nurl = )"[^"]+"', rf'\1"http://127.0.0.1:{self.port}"', text))
            p = tuieval(self.ws, "doctor", check=False, ports=self.port)
        finally:
            mock.stop()
        self.assertIn("✓ local: http://127.0.0.1", p.stdout)
        self.assertIn("✓ wanted (local): served", p.stdout)
        self.assertIn("Ready", p.stdout)


class Detect(unittest.TestCase):
    """Models on servers already running are found: tuieval add, and the guided tuieval init."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.port = free_port()
        empty = os.path.join(self.tmp.name, "nopacks")
        os.makedirs(empty)
        self.mock = Mock(empty, "fixed", port=self.port)    # lists one model, "mock"

    def tearDown(self):
        self.mock.stop()
        self.tmp.cleanup()

    def test_add_finds_the_server_that_has_the_model(self):
        ws = os.path.join(self.tmp.name, "ws")
        tuieval(self.tmp.name, "init", ws)
        out = tuieval(ws, "add", "mock", ports=self.port).stdout
        self.assertIn(f"found mock on the server at http://127.0.0.1:{self.port}", out)
        text = read(os.path.join(ws, "models.toml"))
        self.assertIn(f'[servers.local-{self.port}]\nurl = "http://127.0.0.1:{self.port}"', text)
        self.assertIn(f'server = "local-{self.port}"', text)
        out = tuieval(ws, "add", ports=self.port).stdout                  # nothing new to add
        self.assertIn("already added", out)

    def test_add_without_a_model_lists_and_adds(self):
        ws = os.path.join(self.tmp.name, "ws")
        tuieval(self.tmp.name, "init", ws)
        out = tuieval(ws, "add", "--yes", ports=self.port).stdout
        self.assertIn("1) mock", out)
        self.assertIn("added mock", out)
        out = tuieval(ws, "add", ports=free_port()).stdout                 # nothing running
        self.assertIn("No model server is running", out)
        self.assertIn("tuieval add ~/path/to/model.gguf", out)

    def test_guided_init_reaches_a_first_result(self):
        ws = os.path.join(self.tmp.name, "ws")
        out = tuieval(self.tmp.name, "init", ws, "--yes", ports=self.port).stdout
        self.assertIn("added mock", out)
        self.assertIn("created packs/starter/", out)
        self.assertTrue(os.path.isfile(os.path.join(ws, "results", "smoke", "mock", "starter.json")), out)
        self.assertIn("Next:", out)


class FirstPack(unittest.TestCase):
    """Ways to a first pack without writing YAML by hand: a spreadsheet, a drafting prompt, prompt files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = os.path.join(self.tmp.name, "ws")
        tuieval(self.tmp.name, "init", self.ws)

    def tearDown(self):
        self.tmp.cleanup()

    def test_from_csv(self):
        csv_path = os.path.join(self.tmp.name, "q.csv")
        write(csv_path, 'Question,Answer,Difficulty,Wrong\n"What is 15% of 80?",12,easy,1200|5.33\n'
                        'Capital of Peru?,Lima,easy,Cusco\nRevenue in Q3?,NOT_AVAILABLE,hard,\n')
        out = tuieval(self.ws, "new-pack", "quiz", "--from", csv_path).stdout
        self.assertIn("3 tests from", out)
        self.assertIn(" ok", tuieval(self.ws, "selftest", "quiz").stdout)     # gate fitted to 3 tests
        write(csv_path, "prompt_text,result\nhi,1\n")
        p = tuieval(self.ws, "new-pack", "bad", "--from", csv_path, check=False)
        self.assertIn("needs a question column", p.stderr)

    def test_about_writes_a_drafting_prompt(self):
        out = tuieval(self.ws, "new-pack", "support", "--grader", "reply", "--about", "refunds at a shoe store").stdout
        self.assertIn("DRAFT-PROMPT.md", out)
        text = read(os.path.join(self.ws, "packs", "support", "DRAFT-PROMPT.md"))
        self.assertIn("Topic: refunds at a shoe store", text)
        self.assertIn("must_include", text)                    # the grader's rules and example format
        self.assertIn(" ok", tuieval(self.ws, "selftest", "support").stdout)   # the .md isn't read as tests

    def test_input_file(self):
        from tuieval import packs
        d = os.path.join(self.ws, "packs", "files")
        os.makedirs(os.path.join(d, "prompts"))
        write(os.path.join(d, "prompts", "a.txt"), "What is 2 + 2?\n")
        write(os.path.join(d, "pack.toml"), 'label = "Files"\n')
        write(os.path.join(d, "tests.yaml"), "- {id: a, input_file: prompts/a.txt, expected: 4}\n")
        p = packs.load_pack(d)
        self.assertEqual(p.tests[0]["input"], "What is 2 + 2?")
        before = p.fingerprint
        write(os.path.join(d, "prompts", "a.txt"), "What is 3 + 3?\n")
        self.assertNotEqual(packs.load_pack(d).fingerprint, before)    # editing the file reruns the pack
        write(os.path.join(d, "tests.yaml"), "- {id: a, input_file: ../../models.toml, expected: 4}\n")
        with self.assertRaises(packs.PackError):
            packs.load_pack(d)

    def test_selftest_says_how_to_fix(self):
        tuieval(self.ws, "new-pack", "p")
        path = os.path.join(self.ws, "packs", "p", "tests.yaml")
        write(path, read(path).replace("expected: 12", "expected: 13", 1))
        out = tuieval(self.ws, "selftest", "p", check=False).stdout
        self.assertIn("reference answer fails", out)
        self.assertIn("→ fix the expected answer", out)


def fake_gguf(path, embedding=5120, size=0, kv_heads=4):
    """A GGUF header with just the keys machines.read_gguf reads, padded to size bytes. kv_heads: a
    number, or one per layer (a list, as some GGUFs store it)."""
    import struct
    s = lambda t: struct.pack("<Q", len(t)) + t.encode()  # noqa: E731
    kv = [("general.architecture", 8, "qwen35"), ("qwen35.block_count", 4, 64),
          ("qwen35.embedding_length", 4, embedding), ("qwen35.attention.head_count", 4, 24),
          ("qwen35.attention.head_count_kv", 9 if isinstance(kv_heads, list) else 4, kv_heads),
          ("qwen35.attention.key_length", 4, 256),
          ("qwen35.full_attention_interval", 4, 4), ("qwen35.nextn_predict_layers", 4, 1)]
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for key, t, v in kv:
        out += s(key) + struct.pack("<I", t)
        if t == 8:
            out += s(v)
        elif t == 9:   # an array of uint32
            out += struct.pack("<IQ", 4, len(v)) + b"".join(struct.pack("<I", x) for x in v)
        else:
            out += struct.pack("<I", v)
    with open(path, "wb") as f:
        f.write(out + b"\0" * max(0, size - len(out)))


class FitCheck(unittest.TestCase):
    def test_default_follows_the_ctx_placeholder(self):
        from tuieval import engine
        self.assertTrue(engine.fit_check({"cmd": ["llama-server", "-c", "{ctx}"]}))
        self.assertFalse(engine.fit_check({"cmd": ["other-server", "--model", "{model}"]}))
        self.assertFalse(engine.fit_check({"url": "http://x"}))
        self.assertTrue(engine.fit_check({"cmd": ["other"], "fit_check": True}))
        self.assertFalse(engine.fit_check({"cmd": ["llama", "-c", "{ctx}"], "fit_check": False}))

    def test_own_memory_servers_keep_their_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = os.path.join(tmp, "ws")
            tuieval(tmp, "init", ws)
            fake_gguf(os.path.join(tmp, "m.gguf"), size=1000)
            with open(os.path.join(ws, "models.toml"), "a") as f:
                f.write(f'\n[servers.own]\ncmd = ["own-server", "--model", "{{model}}"]\nport = 18500\nmodel_is_path = true\n'
                        f'\n[[models]]\nlabel = "sized"\nserver = "llama"\nmodel = "{tmp}/m.gguf"\n'
                        f'\n[[models]]\nlabel = "own"\nserver = "own"\nmodel = "{tmp}/m.gguf"\nmax_context = 131072\n')
            code = ("from tuieval import engine; e = engine.Engine(); "
                    "print(e.serving(e.model('own')).ctx, e.serving(e.model('sized')).fit_note != '')")
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                 env=dict(os.environ, TUIEVAL_HOME=ws)).stdout.split()
            self.assertEqual(out, ["131072", "True"])   # own server: its max_context; llama: fit-checked


class WarmTune(unittest.TestCase):
    """tune.tune starting from a tuned model of the same family, on a fake engine whose speed is a
    function of the flags."""
    KNOBS = {"threads": [["-t", "8"], ["-t", "6"]], "ubatch": [["-ub", "512"], ["-ub", "256"], ["-ub", "1024"]],
             "spec": [[], ["--spec-type", "ngram"], ["--spec-type", "draft-mtp"]]}

    def setUp(self):
        import contextlib
        import types
        from tuieval import profiles, tune
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        fake_gguf(f"{d}/new.gguf", size=1000)
        fake_gguf(f"{d}/sib.gguf", size=1100)
        fake_gguf(f"{d}/other.gguf", embedding=4096, size=1000)
        machine = types.SimpleNamespace(id="mac", summary="a Mac", discrete=False)
        eng = types.SimpleNamespace(
            cfg={"servers": {"llama": {"cmd": ["llama"], "tune": self.KNOBS}},
                 "models": [{"label": n, "server": "llama", "model": f"{d}/{n}.gguf"}
                            for n in ("new", "sib", "other")],
                 "defaults": {"request_timeout_ms": 1000}},
            tuning_dir=f"{d}/tuning", root=d, _serving={}, starts=[])
        eng.model = lambda label: {**next(m for m in eng.cfg["models"] if m["label"] == label), "served_name": label}
        eng.serving = lambda m: types.SimpleNamespace(fits=True, ctx=8192, machine=machine, mtp_layers=1,
                                                      identity={"model_bytes": 1000})
        eng.knob_values = lambda mc: {}
        eng.machine = lambda: machine
        eng.server_version = lambda m: "v1"
        eng.gpu_residency_gb = lambda: 24

        @contextlib.contextmanager
        def serve(m, perf_args, log_name):
            eng.starts.append(list(perf_args))
            yield "http://x", {"load_s": 1, "facts": {}}
        eng.serve = serve

        def measure(eng_, m, url, work, *a, **k):
            args = eng.starts[-1]
            fast = {"6": 1, "256": 1, "draft-mtp": 3}
            return tune.Measure(total_s=10 - sum(v for k, v in fast.items() if k in args), texts=["same"])
        self.orig = tune.measure
        tune.measure = measure
        from tuieval import machines
        self.pressure = lambda args: 1   # macOS memory pressure per start's flags: normal
        orig_pressure = machines.memory_pressure_level
        machines.memory_pressure_level = lambda: self.pressure(eng.starts[-1] if eng.starts else [])
        self.addCleanup(setattr, machines, "memory_pressure_level", orig_pressure)
        self.eng, self.tune, self.profiles = eng, tune, profiles
        for label, args in (("sib", ["-t", "6", "-ub", "256", "--spec-type", "draft-mtp"]),
                            ("other", ["-t", "8"])):
            profiles.save("mac", label, {"args": args, "meta": {"method": "seeded", "server_version": "v1"}},
                          eng.tuning_dir)

    def tearDown(self):
        self.tune.measure = self.orig
        self.tmp.cleanup()

    def _swap(self, at_load_mb, while_serving_mb):
        """Fake macOS swap counter: each server start swaps at_load_mb while loading and
        while_serving_mb(args) during the work (args: that start's flags)."""
        from tuieval import machines
        state = {"n": 0, "total": 0}

        def swapped_out_bytes():
            step = state["n"] % 3          # 0: before the start, 1: loaded, 2: after the work
            mb = {0: 0, 1: at_load_mb, 2: while_serving_mb(self.eng.starts[-1]) if step == 2 else 0}[step]
            state["total"] += mb * 2**20
            state["n"] += 1
            return state["total"]
        orig = machines.swapped_out_bytes
        machines.swapped_out_bytes = swapped_out_bytes
        self.addCleanup(setattr, machines, "swapped_out_bytes", orig)

    def test_swapping_while_loading_is_only_noted(self):
        self._swap(at_load_mb=900, while_serving_mb=lambda args: 0)
        p = self.tune.tune(self.eng, "new", use_bench=False)
        self.assertEqual(p["measured"]["server_load_swapped_mb"], 900)
        self.assertEqual(p["meta"]["warnings"], [])

    def test_swapping_every_candidate_shares_is_a_warning(self):
        self._swap(at_load_mb=0, while_serving_mb=lambda args: 800)
        p = self.tune.tune(self.eng, "new", use_bench=False)
        self.assertEqual(p["args"], ["-t", "6", "-ub", "256", "--spec-type", "draft-mtp"])   # tuned as usual
        self.assertIn("800 MB", p["meta"]["warnings"][0])

    def test_a_flag_that_costs_memory_still_wins_with_a_warning(self):
        # -ub 256 is the fastest micro-batch here, and costs 600 MB of swap the defaults don't
        self._swap(at_load_mb=0, while_serving_mb=lambda args: 600 if "256" in args else 100)
        p = self.tune.tune(self.eng, "new", use_bench=False, warm=False)
        self.assertIn("256", p["args"])
        self.assertTrue(any("~500 MB more" in w for w in p["meta"]["warnings"]))

    def test_swap_limit_rules_out_flags_that_cost_memory(self):
        self.eng.cfg["tune"] = {"swap_limit_mb": 256}
        self._swap(at_load_mb=0, while_serving_mb=lambda args: 600 if "256" in args else 100)
        p = self.tune.tune(self.eng, "new", use_bench=False, warm=False)
        self.assertNotIn("256", p["args"])
        self.assertTrue(any("over swap_limit_mb = 256" in n for n in p["meta"]["notes"]))

    def test_critical_memory_pressure_rejects_a_flag(self):
        self.pressure = lambda args: 4 if "draft-mtp" in args else 1
        p = self.tune.tune(self.eng, "new", use_bench=False, warm=False)
        self.assertNotIn("draft-mtp", p["args"])
        self.assertTrue(any("memory pressure turned critical" in n for n in p["meta"]["notes"]))

    def test_projected_time_scores_full_length_answers(self):
        # 10 s for 128 tokens at 16 tok/s: 2 s reading the prompt + 8 s generating
        res = {"error": None, "total_s": 10.0, "ttft_s": 2.0, "reasoning": "", "answer": "x",
               "usage": {"completion_tokens": 128, "prompt_tokens": 50},
               "timings": {"predicted_per_second": 16.0, "prompt_per_second": 25.0}}
        self.eng.cfg["sampling"] = {"temperature": 0}
        self.eng._check = lambda: None
        orig = self.tune._request
        self.tune._request = lambda *a, **k: dict(res)
        self.addCleanup(setattr, self.tune, "_request", orig)
        out = self.tune._measure_once(self.eng, self.eng.model("new"), "http://x", [("p", [])], 128,
                                      lambda *a, **k: None, 60, answer_tokens=1024)
        self.assertEqual((out.total_s, out.projected_s), (10.0, 66.0))   # 2 s + 1024 / 16
        self.assertEqual(out.score("projected"), 66.0)
        self.assertEqual(out.score("total"), 10.0)

    def test_family_ignores_the_mtp_layer(self):
        # one GGUF lists KV heads per layer, none on the MTP (last) layer; the other gives one number
        d = self.tmp.name
        per_layer = [4 if (i + 1) % 4 == 0 else 0 for i in range(63)] + [0]
        fake_gguf(f"{d}/listed.gguf", kv_heads=per_layer)
        self.assertEqual(self.tune.family(f"{d}/listed.gguf"), self.tune.family(f"{d}/new.gguf"))
        fake_gguf(f"{d}/other-shape.gguf", kv_heads=[8 if (i + 1) % 4 == 0 else 0 for i in range(64)])
        self.assertNotEqual(self.tune.family(f"{d}/other-shape.gguf"), self.tune.family(f"{d}/new.gguf"))

    def test_option_index(self):
        opts = self.KNOBS["spec"]
        self.assertEqual(self.tune.option_index(opts, ["-t", "6", "--spec-type", "draft-mtp"]), 2)
        self.assertEqual(self.tune.option_index(opts, ["-t", "6"]), 0)   # off: the empty option
        self.assertIsNone(self.tune.option_index(self.KNOBS["ubatch"], ["-ub", "64"]))

    def test_sibling_is_same_family_only(self):
        sib = self.tune.find_sibling(self.eng, self.eng.model("new"), self.KNOBS, "mac")
        self.assertEqual((sib.label, sib.choices, sib.unmapped), ("sib", {"threads": 1, "ubatch": 1, "spec": 2}, []))
        os.remove(f"{self.tmp.name}/tuning/mac/sib.toml")
        self.assertIsNone(self.tune.find_sibling(self.eng, self.eng.model("new"), self.KNOBS, "mac"))

    def test_warm_start_retries_only_weight_dependent_knobs(self):
        p = self.tune.tune(self.eng, "new", use_bench=False)
        self.assertEqual(p["args"], ["-t", "6", "-ub", "256", "--spec-type", "draft-mtp"])
        self.assertEqual(p["meta"]["warm_start"]["from"], "sib")
        self.assertEqual(p["meta"]["warm_start"]["inherited"], ["threads"])
        self.assertEqual(len(self.eng.starts), 6)   # defaults, sib's flags, 2 other ubatch, 2 other spec
        self.assertTrue(all(s[:2] == ["-t", "6"] for s in self.eng.starts[1:]))   # threads never re-tried

    def test_cold_tunes_every_knob(self):
        p = self.tune.tune(self.eng, "new", use_bench=False, warm=False)
        self.assertNotIn("warm_start", p["meta"])
        self.assertGreater(len(self.eng.starts), 6)

    def test_falls_back_when_inherited_flags_are_slower(self):
        self.profiles.save("mac", "sib", {"args": ["-t", "8", "-ub", "1024"], "meta": {"method": "tuned",
                                          "server_version": "v1"}}, self.eng.tuning_dir)
        orig = self.tune.measure
        self.tune.measure = lambda *a, **k: self.tune.Measure(
            total_s=99 if "1024" in self.eng.starts[-1] else orig(*a).total_s, texts=["same"])
        p = self.tune.tune(self.eng, "new", use_bench=False)
        self.assertNotIn("warm_start", p["meta"])
        self.assertIn("abandoned", " ".join(p["meta"]["notes"]))
        self.assertEqual(p["args"], ["-t", "6", "-ub", "256", "--spec-type", "draft-mtp"])


class ExportPi(unittest.TestCase):
    """export.plan/apply against a temporary presets file and pi models.json."""
    PRESETS = textwrap.dedent("""\
        version = 1

        [*]
        threads         = 8
        ubatch-size     = 256
        cache-reuse     = 256

        [Old]
        model = /nowhere/old.gguf
        temp  = 0.7

        # --- next section's comment ---
        [Keep]
        model = /nowhere/keep.gguf
        """)

    def setUp(self):
        import types
        from tuieval import export
        self.export = export
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        fake_gguf(f"{d}/Old.gguf")
        write(f"{d}/presets.ini", self.PRESETS)
        write(f"{d}/models.json", json.dumps({"providers": {
            "llama": {"apiKey": "SECRET", "modelOverrides": {}}}}, indent=2) + "\n")
        cfg = {"sampling": {"temperature": 1.0, "max_tokens": 16384, "enable_thinking": True},
               "servers": {"llama": {"cmd": ["llama"], "tune": {"cache_reuse": [["--cache-reuse", "256"], []]}},
                           "mlx": {"cmd": ["mlx"]}},
               "models": [{"label": "l", "server": "llama", "model": f"{d}/Old.gguf",
                           "sampling": {"presence_penalty": 1.5}},
                          {"label": "s", "server": "mlx", "model": "~/b"}],
               "export": {"pi": {"presets": f"{d}/presets.ini", "pi_models": f"{d}/models.json"}}}
        eng = types.SimpleNamespace(cfg=cfg)
        eng.model = lambda label: next(m for m in cfg["models"] if m["label"] == label)
        eng.serving = lambda m: types.SimpleNamespace(perf_source="tuned", ctx=98304)
        eng.server_command = lambda m: (["llama", "serve", "-m", m["model"], "--alias", "x", "--port", "1",
                                         "--jinja", "-t", "8", "-ub", "512", "-c", "98304", "--temp", "1.0"],
                                        None, "")
        self.eng, self.d = eng, d

    def tearDown(self):
        self.tmp.cleanup()

    def test_flags_to_keys(self):
        self.assertEqual(self.export.flags_to_keys(["-ub", "512", "--jinja", "--spec-type", "a,b"]),
                         [("ubatch-size", "512"), ("jinja", "true"), ("spec-type", "a,b")])

    def test_set_ini_section_keeps_the_next_sections_comment(self):
        new = self.export.set_ini_section(self.PRESETS, "Old", ["model = /x.gguf"])
        self.assertIn("[Old]\nmodel = /x.gguf\n\n# --- next section's comment ---\n[Keep]", new)
        self.assertNotIn("temp  = 0.7", new)
        self.assertTrue(self.export.set_ini_section(self.PRESETS, "New", ["a = 1"]).endswith("\n\n[New]\na = 1\n"))

    def test_llama_writes_only_what_differs_from_the_common_section(self):
        changes, notes = self.export.plan(self.eng, "l", model_id="Old")
        presets = next(c for c in changes if c.path.endswith("presets.ini")).new
        section = presets.split("[Old]\n")[1].split("\n\n")[0].splitlines()
        self.assertEqual([x.split()[0] for x in section],
                         ["model", "jinja", "ubatch-size", "ctx-size", "temp", "presence-penalty"])
        self.assertIn("presence-penalty = 1.5", presets)
        self.assertNotIn("threads", presets.split("[Old]")[1])
        self.assertTrue(any("cache-reuse" in n for n in notes))   # [*] sets a knob the tuned flags left off
        self.assertTrue(any("don't exist" in n and n.endswith(": Keep") for n in notes))        # names a file that doesn't exist
        pi = json.loads(next(c for c in changes if c.path.endswith("models.json")).new)
        self.assertEqual(pi["providers"]["llama"]["modelOverrides"]["Old"]["contextWindow"], 98304)

    def test_presets_path_is_required(self):
        del self.eng.cfg["export"]["pi"]["presets"]
        with self.assertRaises(self.export.ExportError):
            self.export.plan(self.eng, "l")

    def test_only_llama_servers(self):
        with self.assertRaises(self.export.ExportError):
            self.export.plan(self.eng, "s")

    def test_diff_hides_credentials(self):
        changes, _ = self.export.plan(self.eng, "l", model_id="New")
        diff = next(c for c in changes if c.path.endswith("models.json")).diff()
        self.assertIn('"New"', diff)
        self.assertNotIn("SECRET", diff)

    def test_apply_backs_up_and_refuses_a_file_changed_since(self):
        changes, _ = self.export.plan(self.eng, "l", model_id="New")
        backups = self.export.apply(changes)
        self.assertEqual(len(backups), 2)
        self.assertIn("[New]", read(f"{self.d}/presets.ini"))
        self.assertNotIn("[New]", read(next(b for b in backups if "presets.ini" in b)))
        self.assertEqual(self.export.plan(self.eng, "l", model_id="New")[0], [])   # already exported
        changes, _ = self.export.plan(self.eng, "l", model_id="Other")
        write(f"{self.d}/presets.ini", "edited\n")
        with self.assertRaises(self.export.ExportError):
            self.export.apply(changes)

class Remove(unittest.TestCase):
    TOML = textwrap.dedent("""\
        [servers.llama]
        cmd = ["llama"]

        # first model
        [[models]]
        server = "llama"
        model = "~/a/First-Q4.gguf"

        # second model, with a long comment
        # over two lines
        [[models]]
        label = "second"
        server = "llama"
        model = "~/b.gguf"

        # trailing comment about the next section
        [servers.other]
        url = "http://x"
        """)

    def test_block_removal(self):
        from tuieval import remove
        new = remove.without_block(self.TOML, "second")
        self.assertNotIn("second", new)
        self.assertIn("# first model\n[[models]]", new)
        self.assertIn('model = "~/a/First-Q4.gguf"\n\n# trailing comment about the next section\n[servers.other]', new)
        new = remove.without_block(self.TOML, "first-q4")   # a label derived from the file name
        self.assertNotIn("First-Q4", new)
        self.assertNotIn("# first model", new)
        self.assertIsNone(remove.without_block(self.TOML, "nope"))

    def test_remove_archives_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = os.path.join(tmp, "ws")
            tuieval(tmp, "init", ws)
            with open(os.path.join(ws, "models.toml"), "a") as f:
                f.write('\n[[models]]\nlabel = "gone"\nserver = "llama"\nmodel = "~/gone.gguf"\n')
            os.makedirs(os.path.join(ws, "results", "gone"))
            write(os.path.join(ws, "results", "gone", "x.json"), "{}")
            os.makedirs(os.path.join(ws, "tuning", "mac"))
            write(os.path.join(ws, "tuning", "mac", "machine.toml"), 'id = "mac"\nsummary = "a Mac"\n')
            write(os.path.join(ws, "tuning", "mac", "gone.toml"), 'args = []\n')
            write(os.path.join(ws, "results", "hidden.txt"), "gone\n")
            listed = tuieval(ws, "list").stdout
            self.assertIn("Hidden (1", listed)
            self.assertNotIn("gone", listed.split("Hidden")[0])
            self.assertNotEqual(tuieval(ws, "remove", "--hidden", check=False).returncode, 0)   # asks first
            tuieval(ws, "remove", "--hidden", "--yes")
            self.assertNotIn("gone", read(os.path.join(ws, "models.toml")))
            self.assertNotIn("gone", read(os.path.join(ws, "results", "hidden.txt")))
            self.assertFalse(os.path.exists(os.path.join(ws, "results", "gone")))
            self.assertFalse(os.path.exists(os.path.join(ws, "tuning", "mac", "gone.toml")))
            [kept] = os.listdir(os.path.join(ws, "removed"))
            for rel in ("results/gone/x.json", "tuning/mac/gone.toml", "models.toml-entry.txt", "models.toml.before"):
                self.assertTrue(os.path.exists(os.path.join(ws, "removed", kept, rel)), rel)


if __name__ == "__main__":
    unittest.main()
