"""End-to-end and unit tests. Run from the repo root:  python -m unittest discover -s tests

Uses a temporary workspace and tests/mock_server.py on a free local port; no model or GPU needed.
"""
import json
import os
import re
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
    def __init__(self, packs, mode, port=None):
        self.port = port or free_port()
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


class FirstRun(unittest.TestCase):
    """A new user's first run fails fast with the fix, never hangs or shows a bare Python error."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = os.path.join(self.tmp.name, "ws")
        tuieval(self.tmp.name, "init", self.ws)
        tuieval(self.ws, "new-pack", "first")
        self.port = free_port()
        text = read(os.path.join(self.ws, "models.toml"))   # never look at the machine's real ports
        write(os.path.join(self.ws, "models.toml"), text.replace("[defaults]\n", f"[defaults]\ndetect_ports = [{self.port}]\n", 1)
              .replace('url = "http://127.0.0.1:1234"', f'url = "http://127.0.0.1:{free_port()}"'))

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
                    print(s.query_one("#suites", SelectionList).selected, sorted(s.selected_models), s.tier())
            asyncio.run(go())
        """)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=dict(os.environ, TUIEVAL_HOME=self.ws), timeout=60).stdout.strip()
        self.assertEqual(out, "['first'] ['my-model'] smoke")

    def test_doctor(self):
        tuieval(self.ws, "add", "mock", "--server", "local", "--label", "wanted")
        p = tuieval(self.ws, "doctor", check=False)
        self.assertEqual(p.returncode, 1)
        self.assertIn("✗ local: nothing answers at", p.stdout)
        mock = Mock(os.path.join(self.ws, "packs"), "oracle", port=self.port)   # a server on a detected port
        try:
            out = tuieval(self.ws, "doctor", check=False).stdout
            self.assertIn(f"point url at one that's running: http://127.0.0.1:{self.port}", out)
            text = read(os.path.join(self.ws, "models.toml"))
            write(os.path.join(self.ws, "models.toml"),
                  re.sub(r'(\[servers\.local\]\nurl = )"[^"]+"', rf'\1"http://127.0.0.1:{self.port}"', text))
            p = tuieval(self.ws, "doctor", check=False)
        finally:
            mock.stop()
        self.assertIn("✓ local: http://127.0.0.1", p.stdout)
        self.assertIn("✓ wanted (local): served", p.stdout)
        self.assertIn("Ready", p.stdout)


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
        machine = types.SimpleNamespace(id="mac", summary="a Mac")
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
