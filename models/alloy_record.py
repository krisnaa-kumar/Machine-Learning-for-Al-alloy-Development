
from imports import BaseModel, Dict, List, Any
class AlloyRecord(BaseModel):
    alloy_id: str
    composition_wt: Dict[str, float]
    extrusion_ratio: float | None
    extrusion_rate_mm_s: float | None
    extrusion_temp_C: float | None
    solution_time_h: float | None
    solution_temp_C: float | None
    aging_time_h: float | None
    aging_temp_C: float | None
    quench_temp_C: float | None
    UTS_MPa: float
    elongation_pct: float
    processing_notes: str | None
    source_pdf: str
    source_page: int | None
    paragraph_hash: str

    class Config:
        validate_assignment = True
