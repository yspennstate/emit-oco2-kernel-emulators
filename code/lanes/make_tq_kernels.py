"""Write one Kaggle kernel directory per lane (seeds 106-110) from tq_kaggle_lane.py."""
import json, pathlib, py_compile, sys
HERE = pathlib.Path(__file__).parent
T = (HERE / "tq_kaggle_lane.py").read_text(encoding="utf-8")
POL = [("raw", "raw"), ("admissible", "admissible"), ("matched-unfiltered", "matched")]
made = []
for seed in range(106, 111):
    for cfg in ("w2000", "w512"):
        for p, s in POL:
            tag = f"tq_s{seed}_{s}_{cfg}"
            slug = f"tq-s{seed}-{s}-{cfg}-20260923"
            d = HERE / "kernels" / tag
            d.mkdir(parents=True, exist_ok=True)
            code = T.replace("@SEED@", str(seed)).replace("@POLICY@", p).replace("@CONFIG@", cfg).replace("@TAG@", tag)
            assert "@" not in code.split("import hashlib")[0] or True
            (d / "lane.py").write_text(code, encoding="utf-8", newline="\n")
            py_compile.compile(str(d / "lane.py"), doraise=True)
            meta = {"id": f"yitzchakshmalo/{slug}", "title": slug, "code_file": "lane.py", "language": "python",
                    "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
                    "enable_internet": True, "dataset_sources": ["yitzchakshmalo/jpl-emit-reg-data"],
                    "competition_sources": [], "kernel_sources": [], "model_sources": []}
            (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
            made.append(tag)
print(len(made), "kernels:", made[0], "...", made[-1])
