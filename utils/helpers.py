from imports import yaml, re, uuid, Dict, List, Any, Path

def load_config(path: str|Path="config.yaml") -> dict:
    """Load pipeline settings (thresholds, model names, schema flags) from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)
    
def parse_condition_label(cond: str) -> Dict[str, float]:
    """
    Parse a human-readable condition string from figures/tables into
    unit-explicit, schema-friendly fields. Returns {} if pattern unknown.
    """

    # Example pattern 1 —  "530 °C, 90 min"
    m = re.match(r"(\d+)\s*°C\s*,\s*(\d+)\s*min", cond)
    if m:
        temp  = int(m.group(1))      # °C (already target unit)
        mins  = int(m.group(2))
        hours = round(mins / 60.0, 3)  # store time in hours for consistency

        return {
            "solution_temp_C": temp,   
            "solution_time_h": hours,  
        }

    # Example pattern 2 :"T = 703 K, v = 1.9 mm/s"
    m = re.match(r"T\s*=\s*(\d+)\s*K\s*,\s*v\s*=\s*([\d\.]+)\s*mm/s", cond)
    if m:
        temp_K = int(m.group(1))
        rate   = float(m.group(2))
        temp_C = round(temp_K - 273.15, 1)  # Kelvin -> Celsius to match dataset units
        return {
            "extrusion_temp_C":    temp_C,   # unit-explicit temperature
            "extrusion_rate_mm_s": rate,     # unit-explicit speed
        }

    return {}


def normalize_composition(raw: dict[str,float]
                         | list[dict[str,Any]]) -> list[dict[str,float]]:
    """
    Normalise composition records from LLM/table reads:
      - accept dict or list of dicts
      - drop 'Balance' columns
      - cast numbers to float
      - compute Al as remainder to enforce mass balance to 100 wt.%
    """
    # Accept either a single row or a batch
    raws = raw if isinstance(raw, list) else [raw]

    out = []
    for row in raws:
        # Pull off an optional alloy identifier without mixing it into chemistry
        alloy = row.pop("Alloy", None)

        # Dropping common 'Balance' variants to avoid double-counting Al
        for bad in ("Bal", "Bal.", "Balance", "balance"):
            row.pop(bad, None)

        # Forcing numeric typing (LLMs/tables often yield strings)
        comp = {k: float(v) for k, v in row.items()}

        # Enforcing mass balance: Al = 100 − sum(other elements)
        total = sum(comp.values())
        comp["Al"] = round(100.0 - total, 2)

        # Keeping the source alloy id 
        if alloy is not None:
            comp["_AlloyID"] = alloy

        out.append(comp)
    return out

