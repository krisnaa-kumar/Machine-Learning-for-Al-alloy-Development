
from imports import requests, re, base64, textwrap, argparse, json, re, textwrap, uuid, io, List, Dict, Any

def ask_vision_chart(image_bytes: bytes, cfg: dict) -> List[dict]:
    """
    Sends the cropped chart to your vision LLM and returns the JSON.
    If parsing fails, prints the raw response so you can inspect it.
    """
    prompt = textwrap.dedent("""
      You are a materials‐science extractor. Identify the bar chart of tensile data.
      Read the legend to identify each series name.
      Read the x-axis labels, these are you two processing parameters (these are your "condition" values).
      For each x-axis label, read the number on top of the bars corresponding to "Ultimate Tensile Strength" and the number on top of the bars corresponding to "Elongation". 
      For each condition, read *only* the black numeric label printed on top of each bar. 
      Only read the bar-chart that incldues the strength and elongation.  Do not read any other series (e.g. “Yield strength” or hardness).  
      Return a JSON array of objects, one per condition, in the form:
      [{"condition": <x-axis label string>, "UTS_MPa": <number>, "elongation_pct": <number>}, …]
    """).strip()

    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model":   cfg["vision_model"],
        "stream":  False,
        "messages":[{"role":"user","content":prompt,"images":[b64]}]
    }
    resp = requests.post(cfg["ollama_host"].rstrip("/") + "/api/chat",
                         json=payload, timeout=120)
    resp.raise_for_status()

    raw = resp.json().get("message", {}).get("content","")
    clean = re.sub(r"^```(?:json)?|```$", "", raw).strip()

    if not clean.startswith("["):
        print("\n​--- raw vision‐LLM output ---​")
        print(raw)
        print("​--- end raw output ---​\n")
        raise RuntimeError(
            f"Vision model ({cfg['vision_model']}) did not return JSON. "
            "See raw output above."
        )

    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        print("\n​--- failed JSON parse on ---​")
        print(clean)
        print("​--- end failing JSON ---​\n")
        raise

import re


def ask_vision_table(image_bytes: bytes, cfg: dict) -> dict[str, float]:
    """
    Send the cropped-table image to the vision LLM and return
    a dict mapping element→wt% (mean before ±).
    This handles cases where the LLM returns:
      1) a JSON list of {"Cu":3.0}, {"Li":1.4}, …  
      2) inline arithmetic like "100 - (...)=88.01
    """
    prompt = textwrap.dedent("""
      You are a materials‐science extractor. Below is a table image whose caption begins "Table 1".
                             
      If the first column header is “Alloy”, this is a multi‐row table:
        • Output a JSON **array** of objects, one per row,
          each with "Alloy":<string> plus element→wt% keys.
      Otherwise, output a single JSON object of element→wt% keys.
                             
      **Skip any column whose row cell says “Bal.” or "Balance".**  
      Only return the **mean** before “±” for each **real** element you see.  
      **Do not** attempt any arithmetic or compute Al yourself.
                             
      **If a cell is “–”, record its value as 0.**
                             
      Example multi‐row output:
        [{"Alloy":<number>,"Si":<number>,"Fe":<number>,…}, {"Alloy":"<number>","Si":<number>,…}]
      Example single‐row output:
        {"Cu":<number>,"Li":<number>,…}
    """).strip()

    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model":   cfg["vision_model"],
        "stream":  False,
        "messages":[{"role":"user","content":prompt,"images":[b64]}],
    }
    resp = requests.post(cfg["ollama_host"].rstrip("/") + "/api/chat",
                         json=payload, timeout=120)
    resp.raise_for_status()

    raw = resp.json().get("message",{}).get("content","")
    clean = re.sub(r"^```(?:json)?|```$", "", raw).strip()
    clean = clean.replace("'", '"')
    clean = re.sub(r",\s*([\]}])", r"\1", clean)

    try:
        result = json.loads(clean)
    except json.JSONDecodeError as e:
        print("\n--- raw LLM output ---")
        print(raw)
        print("--- stripped/fixed JSON ---")
        print(clean)
        print(f"--- JSONDecodeError: {e} ---\n")
        raise RuntimeError("Failed to parse JSON from vision-LLM table OCR.")    

    return result

def ask_vision_mech_table(image_bytes: bytes, cfg: dict) -> List[dict]:
    """
    Sends the mechanical-properties table image to the vision LLM
    and returns a list of rows with keys:
      Ratio, Rate_mm_per_min, Tensile_strength_MPa, Elongation_pct
    """
    prompt = textwrap.dedent("""
      You are a materials‐science extractor. Below is an image of a table of mechanical properties.
      Read the columns in this table carefully. 
      **For both UTS and E, read only the mean (the number before the “±”)**.
      Extract every row (skip the header) and return an array of JSON objects with keys, if they are present:
        - Ratio                (number)
        - Rate_mm_per_min      (number; the 'Rate' column)
        - Tensile_strength_MPa (number; from 'UTS [MPa]')
        - Elongation_pct       (number; from 'E[%]')
      No other keys.  Return ONLY valid JSON.
    """).strip()

    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model":   cfg["vision_model"],
        "stream":  False,
        "messages":[{"role":"user","content":prompt,"images":[b64]}]
    }
    resp = requests.post(cfg["ollama_host"].rstrip("/") + "/api/chat",
                         json=payload, timeout=120)
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]
    clean = re.sub(r"^```(?:json)?|```$", "", raw).strip()
    rows = json.loads(clean)

    normalized = []

    for r in rows:
        out = {}
        out["Tensile_strength_MPa"] = float(r.get("Tensile_strength_MPa", r.get("UTS", 0)))
        out["Elongation_pct"]       = float(r.get("Elongation_pct", r.get("E", 0)))

        if "Ratio" in r:
            out["Ratio"] = float(r["Ratio"])

        if "Rate_mm_per_min" in r:
            out["Rate_mm_per_min"] = float(r["Rate_mm_per_min"])
        elif "Rate" in r:
            out["Rate_mm_per_min"] = float(r["Rate"])

        normalized.append(out)
    return normalized

