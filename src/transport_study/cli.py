from __future__ import annotations
import argparse, json, os, shutil, subprocess
from pathlib import Path

def preflight():
    stat=shutil.disk_usage("/workspace"); volume=False
    try:
        data=json.loads(subprocess.check_output(["vast-capabilities"],text=True)); volume=bool(data["instance"]["workspace_is_volume"])
    except Exception: pass
    report={"workspace_persistent":volume,"free_gb":round(stat.free/2**30,2),"required_free_gb":500,"production_ready":volume and stat.free>=500*2**30}
    print(json.dumps(report,indent=2)); return 0 if report["production_ready"] else 2

def main(argv=None):
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    sub.add_parser("preflight")
    b=sub.add_parser("build-manifests"); b.add_argument("--adata",type=Path,required=True); b.add_argument("--out",type=Path,required=True); b.add_argument("--donor-col",default="sample"); b.add_argument("--condition-col",default="cytokine"); b.add_argument("--celltype-col",default="cell_type0528"); b.add_argument("--resamples",type=int,default=20)
    g=sub.add_parser("gate"); g.add_argument("--metrics",type=Path,required=True); g.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    if args.cmd=="preflight": return preflight()
    if args.cmd=="build-manifests":
        import anndata as ad
        from .manifests import build_manifests
        a=ad.read_h5ad(args.adata,backed=None); lock=build_manifests(a,args.out,donor_col=args.donor_col,condition_col=args.condition_col,celltype_col=args.celltype_col,n_resamples=args.resamples); print(json.dumps(lock,indent=2)); return 0 if lock["coverage"]["gate_coverage_possible"] else 4
    if args.cmd=="gate":
        import pandas as pd
        from .evaluation import gate
        frame=pd.read_parquet(args.metrics) if args.metrics.suffix==".parquet" else pd.read_csv(args.metrics); result=gate(frame); args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2)); return 0 if result["passed"] else 3

if __name__=="__main__": raise SystemExit(main())
