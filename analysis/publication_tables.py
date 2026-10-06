"""Derived numerical tables from the aggregate analysis outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import checksum
from .model import VERSION, jaccard


def build(results):
    for log in ("bpi2019", "bpi2017"):
        root = results/log
        base = pd.read_csv(root/"historical_scores.csv").set_index("transition")
        sensitivity = pd.read_csv(root/"sensitivity_transition_risks.csv")
        ports = pd.read_csv(root/"portfolios.csv")
        boundary = []
        positive = base.epsilon.gt(0)
        breakpoints = base.loc[positive, "epsilon_sat"]/base.loc[positive, "epsilon"]
        breakpoints = sorted(set([0., 2.]+breakpoints.loc[breakpoints.between(0,2)].tolist()))
        for k in (3,5,10):
            fixed = set(ports.loc[(ports.method == "Original DRO") & (ports.K == k), "transition"])
            for multiplier in breakpoints:
                scores = base.frequency*np.minimum(2160, base.cvar+multiplier*base.epsilon/.05)
                ordered = scores.rename("score").reset_index().sort_values(["score","transition"],ascending=[False,True])
                selected = set(ordered.transition.head(k))
                outsiders = scores.loc[~scores.index.isin(fixed)]
                margin = float(scores.loc[list(fixed)].min()-outsiders.max())
                boundary.append({"K": k, "radius_multiplier": multiplier, "saturated_candidates": int((base.cvar+multiplier*base.epsilon/.05 >= 2160-1e-7).sum()),
                                 "jaccard_primary": jaccard(fixed,selected), "reference_set_min_margin": margin,
                                 "same_membership": fixed == selected})
        pd.DataFrame(boundary).to_csv(root/"radius_boundary_audit.csv", index=False)
        grid = []
        for label in ("primary","grid_12h","grid_24h"):
            x = sensitivity[sensitivity.configuration.eq(label)].set_index("transition")
            grid.append({"representation": label, "candidates": len(x), "saturated": int(x.saturated.sum()),
                         "median_rho": float(x.rho.median()), "max_abs_risk_change_hours": float(abs(x.risk-base.risk).max()),
                         "max_abs_cvar_change_hours": float(abs(x.cvar-base.cvar).max()),
                         "max_abs_threshold_change_hours": float(abs(x.epsilon_sat-base.epsilon_sat).max()),
                         "changed_saturation_classifications": int((x.saturated != base.saturated).sum())})
        pd.DataFrame(grid).to_csv(root/"grid_comparison.csv", index=False)
        clipping = []
        for label, kappa in (("kappa_0.50",.50),("kappa_0.75",.75),("primary",.90),("kappa_0.95",.95),("kappa_0.99",.99)):
            x = sensitivity[sensitivity.configuration.eq(label)].set_index("transition")
            clipping.append({"kappa": kappa, "active_clipping": int((x.sa_risk < x.risk-1e-8).sum()),
                             "sa_saturated": int((x.sa_risk >= 2160-1e-7).sum())})
        pd.DataFrame(clipping).to_csv(root/"clipping_diagnostics.csv", index=False)
    manifest_path = results/"analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = VERSION
    manifest["code_sha256"] = {p.name: checksum(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    manifest["outputs"] = {str(p.relative_to(results)): checksum(p) for p in sorted(results.rglob("*.csv"))}
    manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=Path("results/availability_analysis"))
    build(p.parse_args().results)
