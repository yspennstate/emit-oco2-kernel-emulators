"""One Kaggle kernel directory per libRadtran lane (seeds 101-110) from lrt_kaggle_lane.py.
usage: python make_lrt_kernels.py [gpu|cpu]   (gpu: request a GPU session; the driver still fits on CPU)"""
import json, pathlib, py_compile, sys
HERE = pathlib.Path(__file__).parent
T = (HERE / "lrt_kaggle_lane.py").read_text(encoding="utf-8")
gpu = (sys.argv[1] if len(sys.argv) > 1 else "gpu") == "gpu"
made = []
for seed in range(101, 111):
    tag = f"lrt_s{seed}_w512"
    slug = f"lrt-s{seed}-w512-20260923"
    d = HERE / "kernels" / tag
    d.mkdir(parents=True, exist_ok=True)
    code = T.replace("@SEED@", str(seed)).replace("@TAG@", tag)
    (d / "lane.py").write_text(code, encoding="utf-8", newline="\n")
    py_compile.compile(str(d / "lane.py"), doraise=True)
    meta = {"id": f"yitzchakshmalo/{slug}", "title": slug, "code_file": "lane.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": ["yitzchakshmalo/libradtran-s2-13band"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    made.append(tag)
print(len(made), "kernels, gpu" if gpu else "kernels, cpu", made[0], "...", made[-1])
