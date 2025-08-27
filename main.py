from imports import (
    argparse, json, re, textwrap, uuid, io, Path,
    torch, np, Image, pd, tqdm, clip, fitz
)
import pytesseract  # OCR guard used to stop XRD/SEM slipping into the LLM path

from models.alloy_record import AlloyRecord           # Typed record → consistent CSV schema
from utils.helpers import load_config, normalize_composition  # YAML cfg + mass-balance Al
from extractors.image_extractor import (
    extract_all_image_blocks,                         # Enumerate all figures (PyMuPDF blocks)
    find_best_match_block,
    crop_and_render_block,                            
    load_reference_embs                               
)
from extractors.table_extractor import (
    find_table1_block,                                # “Table 1” (composition) locator
    find_mechanical_table_block                       # Mechanical props table locator (fallback)
)
from vision.llm_queries import (
    ask_vision_chart,                                 # Vision-LLM reader for bar chart → JSON rows
    ask_vision_table,                                 # Vision-LLM reader for composition table
    ask_vision_mech_table                             # Vision-LLM reader for mechanical table
)
from processing.params_parser import parse_processing_params  # Text-LLM for Methods section → params JSON
from validators.normalizer import validate_and_normalise       # Final schema + unit/logic checks

def looks_like_tensile_chart(pil_img: Image.Image) -> bool:
    """
    Quick OCR gate: only bar charts with UTS/MPa/Elongation labels pass.
    Anything with '2θ', 'Intensity', 'mm', 'µm' is rejected.
    """
    txt = pytesseract.image_to_string(pil_img).lower()
    if any(tok in txt for tok in ("uts", "mpa", "elongation", "%")):
        return True                                      # Accept only tensile-chart tokens
    if any(tok in txt for tok in ("2θ", "intensity", "mm", "µm")):
        return False                                     # Reject common XRD/SEM patterns early
    return False

def top_k_blocks(
    blocks: list[dict],
    ref_embs: np.ndarray,
    clip_model,
    preprocess,
    device: str,
    sim_th: float,
    min_ar: float,
    gray_tol: int,
    k: int = 5
) -> list[dict]:
    """
    Score every block via CLIP and return the top-k above sim_th,
    after applying your aspect-ratio & grayscale filters.
    """
    scored: list[tuple[float, dict]] = []
    for blk in tqdm(blocks, desc="scoring image blocks"):
        pil = blk["img"]
        w, h = pil.size
        # aspect‐ratio filter
        if w / h < min_ar:
            continue                                    # Layout gate: drop tall/non-chart blocks
        arr = np.asarray(pil.convert("L"))  # single‐channel luminance
        if arr.std() < 1.0:             # all-white
            continue                                    # Empty/near-empty blocks → skip

        # embed & score
        tensor = preprocess(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # L2-normalise for cosine similarity
        emb_np = emb.cpu().numpy()  # (1, D)
        sim = float((ref_embs @ emb_np.T).squeeze().max())  # Max cosine vs. reference screenshots
        if sim >= sim_th:
            scored.append((sim, blk))                   # Keeping only blocks likely to be charts

    scored.sort(key=lambda x: x[0], reverse=True)
    return [blk for _, blk in scored[:k]]               # Narrow to few best → fewer LLM calls

def adjusted_bbox_for(blk: dict, use_left_half: bool):
    x0, y0, x1, y1 = blk["bbox"]
    x_mid = (x0 + x1) / 2.0
    if use_left_half:
        return (x0, y0, x_mid, y1)                      # Choosing half with higher CLIP score
    else:
        return (x_mid, y0, x1, y1)


def run(pdf_file: str, out_csv: str, cfg_path: str="config.yaml"):
    cfg      = load_config(cfg_path)                    
    pdf_path = Path(pdf_file)

    # Text‐LLM: processing parameters
    proc = parse_processing_params(str(pdf_path), cfg)  # Deterministic fields (temps/times/rates)

    # Composition Table 1
    comp_block = find_table1_block(pdf_path)
    if comp_block is None:
        raise RuntimeError("Could not locate 'Table 1' (composition) in PDF.")
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(comp_block["page"] - 1)
    x0,y0,x1,y1 = comp_block["bbox"]
    pix = page.get_pixmap(matrix=fitz.Matrix(2,2),
                          clip=fitz.Rect(x0,y0,x1,y1))
    comp_img = pix.tobytes("png")


    # debugging
    with open("debug_table.png","wb") as f:
        f.write(comp_img)
    print("👉 debug_table.png written (composition table)")
    raw_comps = ask_vision_table(comp_img, cfg)         # Vision-LLM reads table cells → JSON
    comp_list    = normalize_composition(raw_comps)     # Cast to float & enforce mass balance

    
    composition_wt = dict(comp_list[0])                 # Single composition dict for chart path
    composition_wt.pop("_AlloyID", None)                # Keep chemistry numeric-only for models
    

    meta = {"pdf": pdf_path.name, "page": comp_block["page"]}  

    # CLIP setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_paths  = [Path(p) for p in cfg["reference_screenshots"]]
    ref_embs, clip_model, preprocess = load_reference_embs(ref_paths, device)  


    # Extracting all image blocks
    blocks = extract_all_image_blocks(pdf_path)         # Full page scan -> candidate figures

    # Finding top‐5 candidates
    candidates = top_k_blocks(
        blocks,
        ref_embs,
        clip_model,
        preprocess,
        device,
        cfg["similarity_threshold"],                     # Cosine cutoff: fewer false positives
        cfg["min_aspect_ratio"],                       
        cfg["grayscale_tol"],
        k=5
    )

    # OCR‐gate + vision‐LLM 
    fig_entries = []
    for blk in candidates:
        # OCR gate on the full block
        if not looks_like_tensile_chart(blk["img"]):
            continue                                   # This is a cheap guard to save expensive LLM calls

        def clip_score(img):
            t = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                e = clip_model.encode_image(t)
                e = e / e.norm(dim=-1, keepdim=True)
            return float((ref_embs @ e.cpu().numpy().T).squeeze().max())

        # split left/right, picking the half CLIP-thinks is closest to ref charts
        left  = blk["img"].crop((0,0, blk["img"].width//2, blk["img"].height))
        right = blk["img"].crop((blk["img"].width//2,0, blk["img"].width, blk["img"].height))
        chosen_half = left if clip_score(left) > clip_score(right) else right  

        # and OCR-gate the half
        if not looks_like_tensile_chart(chosen_half):
            continue
        # crop & render at 2×
        # compute adjusted bbox for chosen half

        use_left = (chosen_half == left)
        clip_bbox = adjusted_bbox_for(blk, use_left)
        # crop & render that half from the original PDF
        img_bytes = crop_and_render_block(
            pdf_path,
            {"page": blk["page"], "bbox": clip_bbox},
            scale=2.0
        )                                             
        # debug
        with open("debug_cropped_block.png", "wb") as f:
            f.write(img_bytes)
        print("👉 debug_cropped_block.png written")

        # asking the vision-LLM
        try:
            fig_entries = ask_vision_chart(img_bytes, cfg)  # Strict schema: {condition, UTS, E}
            break                                           
        except RuntimeError:
            continue                                        
        

    # Fallback to mechanical property table if no bar chart present 
    if not fig_entries:
        print("⚠️  No bar chart detected — falling back to Table 2.")
        mech_block = find_mechanical_table_block(pdf_path)
        if mech_block is None:
            raise RuntimeError("Could not locate mechanical-properties table.")
        page = fitz.open(str(pdf_path)).load_page(mech_block["page"]-1)
        x0,y0,x1,y1 = mech_block["bbox"]
        clip_rect = fitz.Rect(x0, y1, page.rect.x1, page.rect.y1)  # cropping region below caption for composition table
        tbl_pix = page.get_pixmap(matrix=fitz.Matrix(2,2), clip=clip_rect)
        tbl_img = tbl_pix.tobytes("png")
        with open("debug_mech_table.png","wb") as f:
            f.write(tbl_img)
        print("👉 debug_mech_table.png written")

        mech_entries = ask_vision_mech_table(tbl_img, cfg)  # Read UTS/E rows from mechanical table
        recs = []
        for idx, comp in enumerate(comp_list):
            # match mech row by Alloy name
            #  LLM should have returned a field "Alloy" for each row
            # finding the matching mech‐row by Alloy if present
            alloy_id_val = comp.pop("_AlloyID", None)       

            # now finding the matching mechanical row 
            match = next((r for r in mech_entries if r.get("Alloy") == alloy_id_val), None)
            if match is None:
                match = mech_entries[idx] if idx < len(mech_entries) else {}
            
            ratio = match.get("Ratio") or proc.get("extrusion_ratio")
            if "Rate_mm_per_min" in match:
                rate_s = round(match["Rate_mm_per_min"]/60,3)  # Normalising mm/min → mm/s
            else:
                rate_s = proc.get("extrusion_rate_mm_s")
            
            fallback_label = f"ratio={ratio} rate={rate_s}mm/s" 

            # build a sensible string
            if alloy_id_val is None:
                alloy_str = fallback_label
            else:
                try:
                    alloy_str = str(int(float(alloy_id_val)))
                except Exception:
                    alloy_str = str(alloy_id_val)

            recs.append(AlloyRecord(
                alloy_id          =  alloy_str,               
                composition_wt    = comp,                    
                extrusion_ratio   = float(ratio)    if ratio is not None else None,
                extrusion_rate_mm_s = float(rate_s) if rate_s  is not None else None,
                extrusion_temp_C  = proc.get("extrusion_temp_C"),
                solution_time_h   = proc.get("solution_time_h"),
                solution_temp_C   = proc.get("solution_temp_C"),
                aging_time_h      = proc.get("aging_time_h"),
                aging_temp_C      = proc.get("aging_temp_C"),
                quench_temp_C     = proc.get("quench_temp_C"),
                UTS_MPa           = float(match["Tensile_strength_MPa"]),
                elongation_pct    = float(match["Elongation_pct"]),
                processing_notes  = "from Table 2",
                source_pdf        = meta["pdf"],           
                source_page       = mech_block["page"],
                paragraph_hash    = uuid.uuid4().hex[:8]
            ))
        pd.DataFrame([r.model_dump() for r in recs]).to_csv(out_csv, index=False)
        print(f"[✔] extracted {len(recs)} rows from Table 2 → {out_csv}")
        return

    # Final assembly for chart‐based flow
    recs = validate_and_normalise(
        fig_entries,                                      # {condition, UTS, E} from chart
        meta,                                             # source metadata
        composition_wt,                                   # numeric composition (Al as remainder)
        proc                                              # processing params from Methods
    )
    df = pd.DataFrame([r.dict() for r in recs])
    df.to_csv(out_csv, index=False)                       
    print(f"[✔] extracted {len(recs)} rows → {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pdf_file")                            # Input PDF
    p.add_argument("out_csv")                             # Output CSV path
    p.add_argument("--config", default="config.yaml")     # Allows to switch configs without code edits
    args = p.parse_args()
    run(args.pdf_file, args.out_csv, args.config)         

