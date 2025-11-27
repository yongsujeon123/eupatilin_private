#!/usr/bin/env python3
import subprocess, shlex, re
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed

# ================== 설정 ==================
RECEPTOR_DIR   = Path("/home/yongsu/eupatilin/candidate_3")  # *_protein.pdbqt 모여있는 폴더
LIGAND_PATH    = Path("/home/yongsu/eupatilin/eupatilin.pdbqt")  
POCKET_MASTER  = Path("/home/yongsu/eupatilin/candidate_3/ALL_pocket_info.csv")  # 컬럼: pdb_id, center_x..size_z
OUT_ROOT       = Path("/home/yongsu/eupatilin/candidate_3")
SMINA          = Path("/home/yongsu/eupatilin/smina.linux")

NUM_MODES = 10
EXH       = 8
ENERANGE  = 3
N_JOBS    = 8
# ==========================================

OUT_ROOT.mkdir(parents=True, exist_ok=True)

def has_model(pdbqt_path: Path) -> bool:
    if not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
        return False
    with open(pdbqt_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("MODEL"):
                return True
            if i > 50:
                break
    return False

def parse_best_from_pose(pose_path: Path):
    if not pose_path.exists():
        return None
    scores = []
    with open(pose_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "REMARK" in line and "VINA RESULT" in line:
                parts = line.strip().split()
                for tok in parts:
                    try:
                        scores.append(float(tok)); break
                    except:
                        pass
    return min(scores) if scores else None

def load_pocket_master(csv_path: Path):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    need = {"pdb_id","center_x","center_y","center_z","size_x","size_y","size_z"}
    if not need.issubset(df.columns):
        raise ValueError(f"포켓 마스터 컬럼 부족: {sorted(need)} 필요")
    df["pdb_id"] = df["pdb_id"].astype(str).str.upper()
    return df

MASTER = load_pocket_master(POCKET_MASTER)

def lookup_box(pdb_id: str):
    """마스터가 있으면 pdb_id 매칭, 없으면 단일 박스 폴백: center/size를 0,0,0 / 20,20,20 으로."""
    if MASTER is not None:
        r = MASTER.loc[MASTER["pdb_id"] == pdb_id]
        if r.empty:
            raise KeyError(f"pocket box 없음: {pdb_id}")
        r = r.iloc[0]
        center = (float(r["center_x"]), float(r["center_y"]), float(r["center_z"]))
        size   = (float(r["size_x"]),   float(r["size_y"]),   float(r["size_z"]))
        return center, size
    # 폴백(비권장): 모든 리시버에 동일 박스 적용
    return (0.0, 0.0, 0.0), (20.0, 20.0, 20.0)

def infer_pdb_id(path: Path):
    # 예: 7N8T_protein.pdbqt → 7N8T
    m = re.match(r"^([0-9A-Za-z]{4})(?:_[A-Za-z0-9]+)?_(?:protein|receptor|rec)\.pdbqt$", path.name)
    return (m.group(1) if m else path.stem).upper() #대문자로 정제하도록

def run_one(receptor_pdbqt: Path):
    pdb_id = infer_pdb_id(receptor_pdbqt)
    # 생성 파일 이름과 동일한 폴더에 저장
    out_dir = receptor_pdbqt.parent
    out_pose = out_dir / f"{pdb_id}_docked.pdbqt"
    out_log  = out_dir / f"{pdb_id}.log"

    # resume
    if has_model(out_pose):
        score = parse_best_from_pose(out_pose)
        return {"pdb_id": pdb_id, "status": "SKIP_EXIST", "best_affinity": score, "pose": str(out_pose)}

    try:
        center, size = lookup_box(pdb_id)
    except Exception as e:
        return {"pdb_id": pdb_id, "status": f"NO_BOX: {e}", "best_affinity": None, "pose": ""}

    cmd = (
        f'{SMINA} --receptor "{receptor_pdbqt}" --ligand "{LIGAND_PATH}" '
        f'--center_x {center[0]} --center_y {center[1]} --center_z {center[2]} '
        f'--size_x {size[0]} --size_y {size[1]} --size_z {size[2]} '
        f'--num_modes {NUM_MODES} --energy_range {ENERANGE} --exhaustiveness {EXH} '
        f'--out "{out_pose}" --log "{out_log}"'
    )
    try:
        subprocess.run(shlex.split(cmd), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ok = has_model(out_pose)
        score = parse_best_from_pose(out_pose)
        return {"pdb_id": pdb_id, "status": "OK" if ok else "NO_MODEL", "best_affinity": score, "pose": str(out_pose)}
    except subprocess.CalledProcessError as e:
        return {"pdb_id": pdb_id, "status": "ERR", "best_affinity": None, "pose": str(out_pose), "err": e.stderr.decode(errors="ignore")}

def main():
    receptors = sorted(RECEPTOR_DIR.rglob("*.pdbqt"))
    if not receptors:
        raise SystemExit(f"리시버가 없습니다: {RECEPTOR_DIR}")
    rows = Parallel(n_jobs=N_JOBS, prefer="processes")(delayed(run_one)(p) for p in receptors)
    df = pd.DataFrame(rows)
    out_csv = OUT_ROOT / "summary_dir_docking.csv"
    df.to_csv(out_csv, index=False)
    print("✓ summary:", out_csv)
    # 상태 요약
    print(df.groupby("status").size())
    # 상위 10개
    if "best_affinity" in df.columns:
        print(df.dropna(subset=["best_affinity"]).sort_values("best_affinity").head(10)[["pdb_id","best_affinity","pose"]])

if __name__ == "__main__":
    main()