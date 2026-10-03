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

        def measure(eng_, m, url, work, *a):
            args = eng.starts[-1]
            fast = {"6": 1, "256": 1, "draft-mtp": 3}
            return tune.Measure(total_s=10 - sum(v for k, v in fast.items() if k in args), texts=["same"])
        self.orig = tune.measure
        tune.measure = measure
        self.eng, self.tune, self.profiles = eng, tune, profiles
        for label, args in (("sib", ["-t", "6", "-ub", "256", "--spec-type", "draft-mtp"]),
                            ("other", ["-t", "8"])):
            profiles.save("mac", label, {"args": args, "meta": {"method": "seeded", "server_version": "v1"}},
                          eng.tuning_dir)

    def tearDown(self):
        self.tune.measure = self.orig
        self.tmp.cleanup()

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
        self.tune.measure = lambda *a: self.tune.Measure(
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
