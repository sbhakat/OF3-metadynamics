"""Quick Rg comparison: baseline vs metadynamics CIFs."""
import sys
from pathlib import Path
import numpy as np
from biotite.structure.io.pdbx import CIFFile, get_structure


def rg_from_cif(path: Path) -> float:
    cif = CIFFile.read(str(path))
    struct = get_structure(cif, model=1)
    ca = struct[struct.atom_name == "CA"]
    coords = ca.coord
    com = coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - com) ** 2, axis=1))))


def main():
    for label, root in [("baseline", sys.argv[1]), ("metad", sys.argv[2])]:
        cifs = sorted(Path(root).rglob("*.cif"))
        rgs = [rg_from_cif(p) for p in cifs]
        print(f"\n{label} (n={len(rgs)}): mean={np.mean(rgs):.3f} Å, "
              f"std={np.std(rgs):.3f} Å, range=[{min(rgs):.3f}, {max(rgs):.3f}]")
        for p, rg in zip(cifs, rgs):
            print(f"  {p.name}: Rg = {rg:.3f}")


if __name__ == "__main__":
    main()
