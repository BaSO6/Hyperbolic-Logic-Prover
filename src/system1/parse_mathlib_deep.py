# ==========================================
# Filename: src/system1/parse_mathlib_v2.py
# Version: v2.0 (Robust Regex for Lean 4)
# Functionality: 
#   1. Supports theorem parsing with parameters (critical fix)
#   2. Supports protected/noncomputable modifiers
#   3. Supports instance/class extraction
# ==========================================

import os
import re
import json
import argparse
import sys

# ------------------------------------------------------------
# Improved Regex
# ------------------------------------------------------------

# 1. Capture declaration head (Kind + Name)
# Allows prefixes: protected, noncomputable, unsafe, partial, @[simp]...
# Allows Kind: theorem, lemma, def, instance, class, structure, abbrev
# Allows Name: followed by any non-":" characters (i.e., parameter section)
DECL_HEAD_RE = re.compile(
    r'^\s*(?:@\[[^\]]*\]\s*)?'  # Optional attributes @[...], non-capturing
    r'(?:protected\s+|noncomputable\s+|unsafe\s+|partial\s+|scoped\s+)*' # Optional modifiers
    r'\b(theorem|lemma|def|instance|class|structure|abbrev)\s+' # Group 1: Kind
    r'([A-Za-z0-9_\.]+)' # Group 2: Name
    r'(?:\s+.*?)?'       # Optional parameters (ignore content until colon)
    r'\s*:\s*',          # The colon that starts the type signature
    re.MULTILINE
)

# 2. Capture Tactical hints
TACTIC_HINT_RE = re.compile(
    r'\b(rw|simp|have|apply|exact|refine|simp_rw|intro|cases)\b'
)

# 3. Capture referenced theorem names (Capitalized or Dot-separated)
LEMMA_REF_RE = re.compile(
    r'\b([A-Z][A-Za-z0-9_\.]+|[a-z][a-z0-9_]*\.[A-Za-z0-9_\.]+)\b'
)

HEAD_SYMBOLS = {
    '=': 'EQ', '≤': 'LE', '<': 'LT', '≥': 'GE', '>': 'GT',
    '+': 'ADD', '*': 'MUL', '-': 'SUB', '/': 'DIV',
    '∀': 'FORALL', '∃': 'EXISTS', '→': 'IMPLIES', '↔': 'IFF',
    '∧': 'AND', '∨': 'OR', '¬': 'NOT'
}

def detect_head_symbols(type_str):
    heads = set()
    for sym, name in HEAD_SYMBOLS.items():
        if sym in type_str:
            heads.add(name)
    return sorted(heads)

def parse_lean_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    results = {}
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Match declaration
        m = DECL_HEAD_RE.search(line)
        if not m:
            i += 1
            continue

        kind, name = m.groups()
        
        # Attempt to extract Type Signature
        # Simple heuristic: take the segment after the colon until := or end of line
        # Note: This is not perfect, but sufficient for embedding purposes
        rest_of_line = line[m.end():].strip()
        type_sig = rest_of_line.split(":=")[0].strip()
        
        heads = detect_head_symbols(type_sig)

        # Scan subsequent lines for proof dependencies
        used = set()
        # Look ahead up to 50 lines or until next decl
        for j in range(i + 1, min(i + 50, len(lines))):
            curr_line = lines[j]
            
            # Stop if new declaration detected
            if DECL_HEAD_RE.search(curr_line):
                break
                
            # Stop if proof ends (simple heuristic)
            if curr_line.strip() == "" and j > i + 5: # Empty line after some content
                pass 

            # Extract lemma references
            # Only if line looks like a tactic application
            if TACTIC_HINT_RE.search(curr_line) or ":=" in curr_line:
                refs = LEMMA_REF_RE.findall(curr_line)
                for r in refs:
                    # Filter out obvious noise
                    if len(r) > 2 and not r.startswith("haha"): 
                        used.add(r)

        results[name] = {
            "kind": kind,
            "type": type_sig,
            "used_lemmas": sorted(used),
            "head_symbols": heads,
            "file": os.path.basename(path)
        }
        
        i += 1

    return results

def parse_mathlib(root_dir):
    all_data = {}
    print(f"🚀 Scanning {root_dir}...")
    
    file_count = 0
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith(".lean"):
                path = os.path.join(root, fname)
                try:
                    parsed = parse_lean_file(path)
                    all_data.update(parsed)
                    file_count += 1
                    if file_count % 500 == 0:
                        print(f"   ... scanned {file_count} files, found {len(all_data)} decls")
                except Exception as e:
                    # print(f"[WARN] {path}: {e}")
                    pass

    return all_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mathlib_root", required=True)
    parser.add_argument("--output", default="data/proof_local_deps.json")
    args = parser.parse_args()

    data = parse_mathlib(args.mathlib_root)
    print(f"✅ Total Extracted: {len(data)} declarations")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved to: {args.output}")

if __name__ == "__main__":
    main()