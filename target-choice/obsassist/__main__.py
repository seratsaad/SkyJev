"""obsassist command line.

python -m obsassist console   [--program clay_mike_darktime] [--seed 0] [--speed 60] [--port 8765]
python -m obsassist assistant [--console http://127.0.0.1:8765] [--port 8766] [--backend hf|fake|none]
python -m obsassist oracle    --program keck1_lris_tonight --seeds 0-19 [--width 4] [--out results.jsonl]
python -m obsassist etc       --config MIKE-RED --mag 16 --band V --seeing 0.7 --airmass 1.2 --t 1800
python -m obsassist programs
"""

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "console":
        from obsassist.console.server import main as run

        return run(rest)
    if cmd == "assistant":
        from obsassist.assistant.server import main as run

        return run(rest)
    if cmd == "oracle":
        from obsassist.planning.batch import main as run

        return run(rest)
    if cmd == "etc":
        from obsassist.etc_cli import main as run

        return run(rest)
    if cmd == "programs":
        from obsassist.targets import builtin_programs

        for k, p in builtin_programs().items():
            print(f"{k:28s} {p}")
        return 0
    print(f"unknown command {cmd!r}\n{__doc__}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
