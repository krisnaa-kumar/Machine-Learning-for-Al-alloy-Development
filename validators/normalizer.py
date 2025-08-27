
from models.alloy_record import AlloyRecord
from utils.helpers import parse_condition_label

from imports import List, Dict, argparse, json, re, textwrap, uuid, io

# ──────────────────────────────────────────────────────────────────────────────
def validate_and_normalise(entries: List[dict], meta: dict,
                           comp: dict, proc: dict) -> List[AlloyRecord]:

    recs = []
    for e in entries:
        extra = parse_condition_label(e["condition"])

        # seeing if the bar‐chart label itself gave us Kelvin -> °C

        temp_ext= extra.get("extrusion_temp_C")    or proc.get("extrusion_temp_C")
        # same for rate
        extrusion_rate_mm_s = extra.get("extrusion_rate_mm_s") \
                            or proc.get("extrusion_rate_mm_s")

        # the ratio only comes from the text section
        extrusion_ratio     = proc.get("extrusion_ratio")


        recs.append(AlloyRecord(
            alloy_id=e["condition"],
            composition_wt=comp,
            extrusion_ratio=extrusion_ratio,
            extrusion_rate_mm_s=extrusion_rate_mm_s,
            extrusion_temp_C=temp_ext,
            solution_time_h=extra.get("solution_time_h") or proc.get("solution_time_h"),
            solution_temp_C=extra.get("solution_temp_C") or proc.get("solution_temp_C"),
            aging_time_h=None,
            aging_temp_C=None,
            quench_temp_C=proc.get("quench_temp_C"),
            UTS_MPa=e["UTS_MPa"],
            elongation_pct=e["elongation_pct"],
            processing_notes="from figure",
            source_pdf=meta["pdf"],
            source_page=meta["page"],
            paragraph_hash=uuid.uuid4().hex[:8]
        ))
    return recs
