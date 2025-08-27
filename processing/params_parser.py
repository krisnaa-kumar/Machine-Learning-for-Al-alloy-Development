
from imports import fitz, requests, re, json, List, Dict, Any

def parse_processing_params(pdf_path: str, cfg: dict) -> Dict[str,Any]:
    """
    Open the PDF, grab the 'Materials and methods' section, then asking the LLM to extract every processing
    parameter we care about, returning a clean dict.
    """
    #  pulling all text
    doc = fitz.open(pdf_path)
    full_text = "".join(page.get_text("text") + "\n" for page in doc)




    m = re.search(
        r"\d+\.\s*(?:Materials and methods"
        r"|Materials and experimental procedure"
        r"|Materials and experimental details"
        r"|Experimental procedure"
        r"|Experimental methods)"
        r"(.*?)(?=\n\d+\.\s|\Z)",
        full_text, re.IGNORECASE | re.DOTALL
    )
    methods = m.group(1).strip() if m else full_text


    # asking the LLM to extract params
    prompt = f"""
        You are a materials‐science data extractor.  Below is the relevant section from a paper on an Al‐Li alloy extrusion/heat-treatment
        study. 
        In the text there could be two alloys, each with its own "extrusion speed" in m/min. In this case return the extrusion_rate_mm_s (number; convert from m/min to mm/s) for each alloy.
        Please return a _single_ JSON object with these exact keys (use null for any not found):

        - extrusion_ratio            
        - extrusion_rate_mm_s        
        - extrusion_temp_C           (°C)
        - solution_time_h            (hours)
        - solution_temp_C            (°C)
        - aging_time_h               (hours; if specified)
        - aging_temp_C               (°C; if specified)
        - quench_temp_C              (°C; if it says “water‐quenched” return 20)

        If the paper doesn’t mention one of these, set its value to null.  No extra keys.  
        Return _only_ the JSON object - no extra text.

―――
{methods}
―――
"""
    resp = requests.post(
        cfg["ollama_host"].rstrip("/") + "/api/chat",
        json={
            "model": cfg["vision_model"],
            "stream": False,
            "messages": [{"role":"user","content":prompt}]
        },
        timeout=120
    )
    resp.raise_for_status()


    content = resp.json().get("message", {}).get("content", "")
    # striping markdown fences
    cleaned = re.sub(r"^```(?:json)?|```$", "", content).strip()


    try:
        if not cleaned:
            raise ValueError("empty response")
        params = json.loads(cleaned)
    except Exception as e:
        print("\n​--- raw processing‑params LLM output ---​")
        print(content)
        print("--- cleaned JSON candidate ---")
        print(cleaned)
        print(f"⚠️  Warning: failed to parse JSON from processing‑params LLM ({e}); defaulting all to null\n")
        params = {}


    # ensuring numeric/null types
    for k in (
        "extrusion_ratio","extrusion_rate_mm_s","extrusion_temp_C",
        "solution_time_h","solution_temp_C","aging_time_h",
        "aging_temp_C","quench_temp_C"
    ):
        if k not in params:
            params[k] = None
    return params

