from __future__ import annotations
import argparse, json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    d=sub.add_parser("prepare-dong-metadata"); d.add_argument("--raw-adata",type=Path,required=True); d.add_argument("--annotated-adata",type=Path,required=True); d.add_argument("--out",type=Path,required=True)
    b=sub.add_parser("build-manifests"); source=b.add_mutually_exclusive_group(required=True); source.add_argument("--adata",type=Path); source.add_argument("--metadata",type=Path); b.add_argument("--out",type=Path,required=True); b.add_argument("--donor-col",default="sample"); b.add_argument("--condition-col",default="cytokine"); b.add_argument("--celltype-col",default="cell_type0528"); b.add_argument("--resamples",type=int,default=20)
    g=sub.add_parser("gate"); g.add_argument("--metrics",type=Path,required=True); g.add_argument("--out",type=Path,required=True)
    r=sub.add_parser("run"); r.add_argument("--manifests",type=Path,required=True); r.add_argument("--metadata",type=Path,required=True); r.add_argument("--raw-adata",type=Path,required=True); r.add_argument("--ckpt",type=Path,required=True); r.add_argument("--genes",type=Path,required=True); r.add_argument("--out",type=Path,required=True); r.add_argument("--device",default="cuda"); r.add_argument("--limit",type=int)
    args=p.parse_args(argv)
    if args.cmd=="prepare-dong-metadata":
        from .dong_schema import prepare_dong_metadata, write_dong_metadata
        metadata=prepare_dong_metadata(args.raw_adata,args.annotated_adata); report=write_dong_metadata(metadata,args.out); print(json.dumps(report,indent=2)); return 0
    if args.cmd=="build-manifests":
        if args.metadata:
            from .dong_schema import read_dong_metadata
            from .manifests import build_manifests_from_metadata
            metadata=read_dong_metadata(args.metadata); lock=build_manifests_from_metadata(metadata.obs,metadata.genes,args.out,donor_col=args.donor_col,condition_col=args.condition_col,celltype_col=args.celltype_col,n_resamples=args.resamples)
        else:
            import anndata as ad
            from .manifests import build_manifests
            a=ad.read_h5ad(args.adata,backed=None); lock=build_manifests(a,args.out,donor_col=args.donor_col,condition_col=args.condition_col,celltype_col=args.celltype_col,n_resamples=args.resamples)
        print(json.dumps(lock,indent=2)); return 0 if lock["coverage"]["gate_coverage_possible"] else 4
    if args.cmd=="gate":
        import pandas as pd
        from .evaluation import gate
        frame=pd.read_parquet(args.metrics) if args.metrics.suffix==".parquet" else pd.read_csv(args.metrics); result=gate(frame); args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2)); return 0 if result["passed"] else 3
    if args.cmd=="run":
        from .run import run_study
        run_study(args.manifests,args.metadata,args.ckpt,args.genes,args.raw_adata,args.out,device=args.device,limit=args.limit); return 0

if __name__=="__main__": raise SystemExit(main())
