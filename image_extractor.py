
from imports import Image, fitz, io, List, Dict, Any, Path, np, torch, tqdm, clip

# ──────────────────────────────────────────────────────────────────────────────
def load_reference_embs(ref_paths: List[Path], device="cpu") -> tuple[np.ndarray, Any, Any]:
    """
    Load reference screenshots, compute CLIP embeddings with openai/clip,
    and return (N_ref×D) array plus the CLIP model and preprocess_fn.
    """
    model, preprocess = clip.load("ViT-B/32", device=device)
    embs = []
    for p in ref_paths:
        img = Image.open(p).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        embs.append(emb.cpu().numpy())
    return np.vstack(embs), model, preprocess


def extract_all_image_blocks(pdf_path: Path) -> List[dict]:
    """
    For every page in the PDF, returning a list of dicts:
      { "page": int,
        "bbox": (x0,y0,x1,y1),
        "img": PIL.Image }    decoding raw block image bytes into PIL
    """
    doc = fitz.open(str(pdf_path))
    out = []
    for pno, page in enumerate(doc, start=1):
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 1:
                continue
            x0, y0, x1, y1 = b["bbox"]
            img_data = b.get("image")     
            if not isinstance(img_data, (bytes, bytearray)):
               
                continue
            # decoding into a PIL.Image
            try:
                pil = Image.open(io.BytesIO(img_data)).convert("RGB")
            except Exception:
               
                continue

            out.append({
                "page": pno,
                "bbox": (x0, y0, x1, y1),
                "img":  pil
            })
    return out


def find_best_match_block(
    blocks: List[dict],
    ref_embs: np.ndarray,
    clip_model,
    preprocess,            
    device: str,
    similarity_threshold: float,
    min_aspect_ratio: float,
    grayscale_tol: int
) -> dict | None:
    """
    Picking the block whose CLIP embedding best matches one of our references,
    but only after filtering out:
      - grayscale blocks (SEM images)
    """
    best_score, best_blk = -1.0, None

    for blk in tqdm(blocks, desc="scoring image blocks"):
        pil = blk["img"]
        w, h = pil.size

        if w / h < min_aspect_ratio:
            continue

        arr = np.asarray(pil)
        if arr.ndim == 3:
            if (np.allclose(arr[:,:,0], arr[:,:,1], atol=grayscale_tol) and
                np.allclose(arr[:,:,0], arr[:,:,2], atol=grayscale_tol)):
                continue

        tensor = preprocess(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        emb_np = emb.cpu().numpy()

        sims = (ref_embs @ emb_np.T).squeeze()
        score = float(sims.max())
        if score > best_score:
            best_score, best_blk = score, blk

    # applying similarity threshold
    if best_blk is None or best_score < similarity_threshold:
        return None
    return best_blk


# ──────────────────────────────────────────────────────────────────────────────
def crop_and_render_block(pdf_path: Path, blk: dict, scale: float = 2.0) -> bytes:
    """
    Given the winning block, re-render that region at `scale`× and returning raw PNG bytes.
    """
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(blk["page"]-1)
    x0,y0,x1,y1 = blk["bbox"]

    mat  = fitz.Matrix(scale, scale)
    clip = fitz.Rect(x0, y0, x1, y1)
    pix  = page.get_pixmap(matrix=mat, clip=clip)

    return pix.tobytes("png")
