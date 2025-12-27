# --*-- conding:utf-8 --*--
# @time:12/27/25 16:29
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:find_b_segment_in_a.py

# find_b_segment_in_a.py
from pathlib import Path

AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    # common variants
    "MSE":"M",
}

def is_atom(line: str) -> bool:
    return line.startswith("ATOM") or line.startswith("HETATM")

def resname(line: str) -> str:
    return line[17:20].strip()

def chain_id(line: str) -> str:
    return (line[21].strip() or " ")

def resseq(line: str) -> int:
    return int(line[22:26].strip())

def atom_name(line: str) -> str:
    return line[12:16].strip()

def extract_chain_residues(pdb_path: Path, chain: str):
    """
    Return ordered list of (resseq, one_letter) for residues that have CA atom in ATOM records.
    """
    seen = set()
    out = []
    for ln in pdb_path.read_text().splitlines():
        if not is_atom(ln):
            continue
        if not ln.startswith("ATOM"):
            continue
        if chain_id(ln) != chain:
            continue
        if atom_name(ln) != "CA":
            continue
        rseq = resseq(ln)
        if rseq in seen:
            continue
        seen.add(rseq)
        aa1 = AA3_TO_1.get(resname(ln), "X")
        out.append((rseq, aa1))
    return out

def seq_and_index(res_list):
    seq = "".join([aa for _, aa in res_list])
    idx = [rseq for rseq, _ in res_list]
    return seq, idx

def main():
    pdb = Path("RCSB_KRAS/4lpk.pdb")
    A = extract_chain_residues(pdb, "A")
    B = extract_chain_residues(pdb, "B")

    seqA, idxA = seq_and_index(A)
    seqB, idxB = seq_and_index(B)

    # build a map from residue number to position in B sequence
    b_pos = {r:i for i,(r,_) in enumerate(B)}

    b_start, b_end = 55, 64
    if b_start not in b_pos or b_end not in b_pos:
        raise SystemExit(f"Chain B does not contain full CA coverage for {b_start}-{b_end} in {pdb}")

    i0 = b_pos[b_start]
    i1 = b_pos[b_end]
    if i1 < i0:
        raise SystemExit("Unexpected ordering in chain B residues")

    seg = seqB[i0:i1+1]
    print("Chain B segment 55-64:", seg)

    # find all occurrences in A
    hits = []
    start = 0
    while True:
        j = seqA.find(seg, start)
        if j == -1:
            break
        # map to A residue numbers (by CA order)
        a_res_start = idxA[j]
        a_res_end = idxA[j + len(seg) - 1]
        hits.append((j, a_res_start, a_res_end))
        start = j + 1

    if not hits:
        print("No exact sequence window match found in chain A.")
        print("This suggests B is not a direct sequence-position copy you can map by window matching.")
        return

    print("Matches in chain A:")
    for j, rs, re in hits:
        print(f"  - seqA_pos={j}  A_res_range={rs}-{re}")

if __name__ == "__main__":
    main()
