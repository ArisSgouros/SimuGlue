from __future__ import annotations
from simuglue.template import template_function
import argparse

def main():
    p = argparse.ArgumentParser(description="template_function: exports an input string")
    p.add_argument("-msg", "--msg", default="no msg")

    args = p.parse_args()
    msg_in = args.msg
    msg_proc = template_function(msg_in)

    print("template_cli:")
    print(f"input msg: {msg_in}")
    print(f"output msg: {msg_proc}")
