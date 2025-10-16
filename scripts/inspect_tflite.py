#!/usr/bin/env python3
# inspect_tflite.py
# Usage:
#   python inspect_tflite.py path/to/model.tflite
#   python inspect_tflite.py path/to/model.tflite --quant

import argparse
import sys
from pathlib import Path

def load_interpreter(model_path):
    try:
        import tensorflow as tf  # tensorflow>=2.x
        return tf.lite.Interpreter(model_path=str(model_path))
    except Exception:
        try:
            from tflite_runtime.interpreter import Interpreter
            return Interpreter(model_path=str(model_path))
        except Exception as e:
            print(f"[ERROR] Unable to import TensorFlow or tflite_runtime: {e}", file=sys.stderr)
            sys.exit(2)

def fmt_shape(shape):
    try:
        return list(shape)
    except Exception:
        return shape

def print_tensor_block(title, tensors, show_quant=False):
    print(title)
    if not tensors:
        print("  (none)")
        return
    for i, d in enumerate(tensors):
        name  = d.get("name")
        idx   = d.get("index")
        shape = fmt_shape(d.get("shape"))
        dtype = d.get("dtype")
        print(f"  [{i}] name={name} index={idx} shape={shape} dtype={dtype}")
        if show_quant:
            qp = d.get("quantization_parameters") or {}
            scales = qp.get("scales", [])
            zps    = qp.get("zero_points", [])
            if (hasattr(scales, "__len__") and len(scales) > 0) or (hasattr(zps, "__len__") and len(zps) > 0):
                print(f"       quant: scales={list(scales)} zero_points={list(zps)}")

def main():
    ap = argparse.ArgumentParser(description="Inspect TFLite model IO shapes/dtypes.")
    ap.add_argument("model", type=Path, help=".tflite file path")
    ap.add_argument("--quant", action="store_true", help="show quantization params (scales/zero_points)")
    args = ap.parse_args()

    if not args.model.exists():
        print(f"[ERROR] File not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    interpreter = load_interpreter(args.model)
    try:
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"[ERROR] allocate_tensors() failed: {e}", file=sys.stderr)
        sys.exit(3)

    in_details  = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    print(f"Model: {args.model.resolve()}")
    print_tensor_block("=== INPUTS ===", in_details, show_quant=args.quant)
    print_tensor_block("=== OUTPUTS ===", out_details, show_quant=args.quant)

    # Ringkasan ringkas (baris terakhir enak untuk grep)
    ins  = ", ".join([f"{fmt_shape(d['shape'])}" for d in in_details])
    outs = ", ".join([f"{fmt_shape(d['shape'])}" for d in out_details])
    dtypes_in  = ", ".join([str(d["dtype"]) for d in in_details])
    dtypes_out = ", ".join([str(d["dtype"]) for d in out_details])
    print("\nSUMMARY:")
    print(f"  inputs : {len(in_details)} | shapes: {ins} | dtypes: {dtypes_in}")
    print(f"  outputs: {len(out_details)} | shapes: {outs} | dtypes: {dtypes_out}")

if __name__ == "__main__":
    main()
