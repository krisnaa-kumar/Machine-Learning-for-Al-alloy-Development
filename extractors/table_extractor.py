
from imports import fitz, re, Path

def find_table1_block(pdf_path: Path) -> dict | None:
    """
    Composition table:
    Looking for a text‐block which first line begins 'Table 1',
    then returning a crop region from just under that block
    down to the bottom of the page.
    """
    doc = fitz.open(str(pdf_path))
    for pno, page in enumerate(doc, start=1):
        # pulling out all text‐blocks
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            # reconstructing the first line's text
            first_line = "".join(span["text"] 
                                 for span in block["lines"][0]["spans"])
            if first_line.strip().startswith("Table 1"):
                x0, y0, x1, y1 = block["bbox"]
                clip = fitz.Rect(x0, y1, page.rect.x1, page.rect.y1)
                return {"page": pno, "bbox": (clip.x0, clip.y0, clip.x1, clip.y1)}
    return None

#### For papers with no bar-chart only mechanical property table of UTS and elongation
def find_mechanical_table_block(pdf_path: Path) -> dict | None:

    doc = fitz.open(str(pdf_path))
    for pno, page in enumerate(doc, start=1):
        txt = page.get_text("text")
        if "Table 2" in txt and "Tensile Strength" in txt:
            for block in page.get_text("dict")["blocks"]:
                if block["type"] == 0:
                    text = "".join(span["text"] for line in block["lines"] for span in line["spans"])
                    if text.strip().startswith("Table 2"):  
                        x0,y0,x1,y1 = block["bbox"]
                        return {"page": pno, "bbox": (x0, y0, x1, y1)}
    return None

