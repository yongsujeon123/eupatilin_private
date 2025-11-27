from pymol import cmd

def compare_docking_interactions(exp_obj, dockprot_obj, docklig_obj, cutoff=4.0):
    """
    Compare interacting residues between experimental ligand and docking ligands.
    """
    cutoff = float(cutoff)

    # 실험 리간드가 단백질과 상호작용한 잔기
    exp_contacts = set()
    for a, b in cmd.get_pairs(f"{exp_obj} within {cutoff} of {dockprot_obj}"):
        model = cmd.get_model(f"id {b[1]}")
        if model.atom:
            atom = model.atom[0]
            exp_contacts.add((atom.resn, atom.resi, atom.chain))

    # 도킹 리간드가 단백질과 상호작용한 잔기
    dock_contacts = set()
    for a, b in cmd.get_pairs(f"{docklig_obj} within {cutoff} of {dockprot_obj}"):
        model = cmd.get_model(f"id {b[1]}")
        if model.atom:
            atom = model.atom[0]
            dock_contacts.add((atom.resn, atom.resi, atom.chain))

    # 출력
    print("\n===== Interaction Comparison =====")
    print("Experimental ligand contacts:")
    for r in sorted(exp_contacts):
        print("  ", r)

    print("\nDocking ligand contacts:")
    for r in sorted(dock_contacts):
        print("  ", r)

    print("\nCommon residues:")
    for r in sorted(exp_contacts & dock_contacts):
        print("  ", r)

    print("=================================\n")

# 반드시 필요! PyMOL에 명령 등록
cmd.extend("compare_docking_interactions", compare_docking_interactions)
