"""Write an Ollama Modelfile for the tuned GGUF file.

    ollama pull qwen3:1.7b
    python scripts/make_modelfile.py --gguf models/pii-qwen3-1.7b-q8_0.gguf
    ollama create pii-qwen3-lora -f Modelfile

A GGUF file imported on its own may be served without the right chat template.
The tuned model uses the same chat format as its base model, so this script
takes the base model's Modelfile from Ollama and only swaps the weights.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_modelfile(base_modelfile: str, gguf_path: str) -> str:
    lines = []
    replaced = False
    for line in base_modelfile.splitlines():
        if line.strip().upper().startswith("FROM ") and not replaced:
            lines.append(f"FROM {gguf_path}")
            replaced = True
        elif line.strip().upper().startswith("PARAMETER TEMPERATURE"):
            continue
        else:
            lines.append(line)
    if not replaced:
        raise ValueError("the base Modelfile has no FROM line")
    lines.append("PARAMETER temperature 0")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Create a Modelfile that serves the tuned GGUF with the base model's chat template.")
    p.add_argument("--gguf", required=True, help="path to the exported GGUF file")
    p.add_argument("--base", default="qwen3:1.7b", help="Ollama model whose template and parameters are reused")
    p.add_argument("--out", default="Modelfile")
    args = p.parse_args(argv)

    gguf = Path(args.gguf)
    if not gguf.is_file():
        print(f"ERROR: {gguf} not found. Download the GGUF file from your Hub repository first.")
        return 1
    try:
        shown = subprocess.run(["ollama", "show", args.base, "--modelfile"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        print("ERROR: the 'ollama' command was not found. Install Ollama first.")
        return 1
    if shown.returncode != 0:
        print(f"ERROR: could not read the Modelfile of '{args.base}'. Run `ollama pull {args.base}` first.\n{shown.stderr.strip()}")
        return 1
    try:
        content = build_modelfile(shown.stdout, str(gguf.resolve()))
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    Path(args.out).write_text(content, encoding="utf-8")
    print(f"wrote {args.out}. Next: ollama create pii-qwen3-lora -f {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
