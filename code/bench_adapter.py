"""Load a papers 3-5 bench corpus in the data-dictionary form the paper-2 drivers expect.

The drivers (rmt_krr.py, kf_kernels.py, kf_features.py) read D["Xtr"|"Xva"|"Xte"|"Ztr"|"Zva"|"Zte"],
D["phys_err"](Zp, split), D["tag"], D["names"], D["Yobj"], D["ynorm2"]; bench_data._pack provides all of
them under the name "err" instead of "phys_err". This module adds the alias and the per-corpus network
defaults that bench_run.py uses, so a bench lane in a driver trains the same network family the bench
engine trains.

    D, hp = bench_adapter.load(args)      # args: corpus, seed, ntrain, nu, beta, well_name, rank, lowfi
"""
import bench_data


def add_args(p):
    p.add_argument("--corpus", default="", help="bench corpus when --problem bench: climsim, pkanrtm, trl2d, well, burgers, advection, darcy")
    p.add_argument("--nu", type=float, default=0.1, help="Burgers viscosity")
    p.add_argument("--beta", type=float, default=1.0, help="Darcy coefficient contrast")
    p.add_argument("--well_name", default="active_matter")
    p.add_argument("--rank", type=int, default=0, help="TRL2D / Well PCA rank (default 256)")
    p.add_argument("--lowfi", type=int, default=0, help="pKANrtm: 1 = the 6S low-fidelity mean as an input")


def load(args):
    c = args.corpus
    if c == "climsim":
        D = bench_data.climsim(args.seed, args.ntrain or 100000, nval=20000, ntest=20000); hp = dict(EPOCHS=60, WIDTH=512, DEPTH=4, BS=1024)
    elif c == "pkanrtm":
        D = bench_data.pkanrtm(args.seed, args.ntrain, args.lowfi); hp = dict(EPOCHS=120, WIDTH=384, DEPTH=4, BS=1024)
    elif c == "trl2d":
        D = bench_data.trl2d(args.seed, args.ntrain, rank_in=args.rank or 256, rank_out=args.rank or 256); hp = dict(EPOCHS=200, WIDTH=512, DEPTH=4, BS=256)
    elif c == "well":
        import well_data
        D = well_data.well2d(args.well_name, args.seed, args.ntrain, rank_in=args.rank or 256, rank_out=args.rank or 256); hp = dict(EPOCHS=200, WIDTH=512, DEPTH=4, BS=256)
    elif c == "burgers":
        D = bench_data.burgers(args.nu, args.seed, args.ntrain or 8000); hp = dict(EPOCHS=200, WIDTH=512, DEPTH=4, BS=256)
    elif c == "advection":
        D = bench_data.advection(args.seed, args.ntrain or 20000); hp = dict(EPOCHS=200, WIDTH=512, DEPTH=4, BS=512)
    elif c == "darcy":
        D = bench_data.darcy(args.beta, args.seed, args.ntrain or 8000); hp = dict(EPOCHS=200, WIDTH=512, DEPTH=4, BS=256)
    else:
        raise SystemExit(f"unknown bench corpus {c!r}")
    D["phys_err"] = D["err"]
    return D, hp
