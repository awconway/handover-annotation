from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
W = f"{{{W_NS}}}"
REL = f"{{{REL_NS}}}"
CT = f"{{{CONTENT_TYPES_NS}}}"
NS = {"w": W_NS}

TARGET_HEADERS = [
    "Checklist item",
    "TP",
    "FP",
    "FN",
    "TN",
    "Precision",
    "Recall",
    "F1",
]

DEFAULT_PAGE_SIZE = {
    W + "w": "12240",
    W + "h": "15840",
}
DEFAULT_PAGE_MARGINS = {
    W + "top": "1440",
    W + "right": "1440",
    W + "bottom": "1440",
    W + "left": "1440",
    W + "header": "720",
    W + "footer": "720",
    W + "gutter": "0",
}
PPR_CHILD_ORDER = [
    "pStyle",
    "keepNext",
    "keepLines",
    "pageBreakBefore",
    "framePr",
    "widowControl",
    "numPr",
    "suppressLineNumbers",
    "pBdr",
    "shd",
    "tabs",
    "suppressAutoHyphens",
    "kinsoku",
    "wordWrap",
    "overflowPunct",
    "topLinePunct",
    "autoSpaceDE",
    "autoSpaceDN",
    "bidi",
    "adjustRightInd",
    "snapToGrid",
    "spacing",
    "ind",
    "contextualSpacing",
    "mirrorIndents",
    "suppressOverlap",
    "jc",
    "textDirection",
    "textAlignment",
    "textboxTightWrap",
    "outlineLvl",
    "divId",
    "cnfStyle",
    "rPr",
    "sectPr",
    "pPrChange",
]
PPR_ORDER = {W + name: index for index, name in enumerate(PPR_CHILD_ORDER)}

ET.register_namespace("w", W_NS)
ET.register_namespace("", REL_NS)


def text_content(element: ET.Element) -> str:
    return "".join(node.text or "" for node in element.findall(".//w:t", NS)).strip()


def direct_children(element: ET.Element, tag: str) -> list[ET.Element]:
    return [child for child in list(element) if child.tag == tag]


def dedupe_child_tags(parent: ET.Element, tag: str, keep: str = "last") -> None:
    matches = direct_children(parent, tag)
    if len(matches) <= 1:
        return

    keeper = matches[-1] if keep == "last" else matches[0]
    for match in matches:
        if match is not keeper:
            parent.remove(match)


def merge_duplicate_paragraph_properties(root: ET.Element) -> int:
    fixed = 0
    for paragraph in root.findall(".//w:p", NS):
        pprs = direct_children(paragraph, W + "pPr")
        if len(pprs) <= 1:
            continue

        first = pprs[0]
        for extra in pprs[1:]:
            for child in list(extra):
                for existing in direct_children(first, child.tag):
                    first.remove(existing)
                first.append(child)
            paragraph.remove(extra)
            fixed += 1

        children = list(paragraph)
        if children and children[0] is not first:
            paragraph.remove(first)
            paragraph.insert(0, first)

    return fixed


def paragraph_properties(paragraph: ET.Element) -> ET.Element:
    pprs = direct_children(paragraph, W + "pPr")
    if pprs:
        return pprs[0]

    ppr = ET.Element(W + "pPr")
    paragraph.insert(0, ppr)
    return ppr


def set_paragraph_alignment(paragraph: ET.Element, alignment: str) -> None:
    ppr = paragraph_properties(paragraph)
    for jc in direct_children(ppr, W + "jc"):
        ppr.remove(jc)

    jc = ET.Element(W + "jc")
    jc.set(W + "val", alignment)
    ppr.append(jc)


def normalize_duplicate_alignments(root: ET.Element) -> int:
    fixed = 0
    for ppr in root.findall(".//w:pPr", NS):
        count_before = len(direct_children(ppr, W + "jc"))
        dedupe_child_tags(ppr, W + "jc", keep="last")
        if count_before > 1:
            fixed += 1
    return fixed


def normalize_paragraph_property_order(root: ET.Element) -> int:
    fixed = 0
    for ppr in root.findall(".//w:pPr", NS):
        children = list(ppr)
        sorted_children = sorted(
            children,
            key=lambda child: (PPR_ORDER.get(child.tag, len(PPR_ORDER)), children.index(child)),
        )
        if children == sorted_children:
            continue

        for child in children:
            ppr.remove(child)
        for child in sorted_children:
            ppr.append(child)
        fixed += 1

    return fixed


def ensure_child(parent: ET.Element, tag: str, attrs: dict[str, str]) -> int:
    if direct_children(parent, tag):
        return 0

    child = ET.Element(tag)
    for key, value in attrs.items():
        child.set(key, value)
    parent.append(child)
    return 1


def normalize_section_properties(root: ET.Element) -> int:
    fixed = 0
    for section in root.findall(".//w:sectPr", NS):
        fixed += ensure_child(section, W + "pgSz", DEFAULT_PAGE_SIZE)
        fixed += ensure_child(section, W + "pgMar", DEFAULT_PAGE_MARGINS)
        fixed += ensure_child(section, W + "cols", {W + "space": "720"})
    return fixed


def first_paragraph(block: ET.Element) -> ET.Element | None:
    if block.tag == W + "p":
        return block
    if block.tag == W + "tbl":
        return block.find(".//w:p", NS)
    return None


def last_paragraph(block: ET.Element) -> ET.Element | None:
    if block.tag == W + "p":
        return block
    if block.tag == W + "tbl":
        paragraphs = block.findall(".//w:p", NS)
        return paragraphs[-1] if paragraphs else None
    return None


def first_content_index(paragraph: ET.Element) -> int:
    children = list(paragraph)
    if children and children[0].tag == W + "pPr":
        return 1
    return 0


def normalize_body_bookmarks(root: ET.Element) -> int:
    body = root.find("./w:body", NS)
    if body is None:
        return 0

    fixed = 0
    index = 0
    while index < len(list(body)):
        children = list(body)
        child = children[index]
        if child.tag not in {W + "bookmarkStart", W + "bookmarkEnd"}:
            index += 1
            continue

        target: ET.Element | None = None
        if child.tag == W + "bookmarkStart":
            for next_child in children[index + 1 :]:
                target = first_paragraph(next_child)
                if target is not None:
                    target.insert(first_content_index(target), child)
                    break
        else:
            for previous_child in reversed(children[:index]):
                target = last_paragraph(previous_child)
                if target is not None:
                    target.append(child)
                    break

        body.remove(child)
        fixed += 1

    return fixed


def table_rows(table: ET.Element) -> list[ET.Element]:
    return table.findall("./w:tr", NS)


def row_cells(row: ET.Element) -> list[ET.Element]:
    return row.findall("./w:tc", NS)


def is_checklist_table(table: ET.Element) -> bool:
    rows = table_rows(table)
    if not rows:
        return False

    header = [text_content(cell) for cell in row_cells(rows[0])]
    return header == TARGET_HEADERS


def normalize_checklist_table(table: ET.Element) -> int:
    fixed = 0
    for row in table_rows(table):
        for column_index, cell in enumerate(row_cells(row)):
            alignment = "left" if column_index == 0 else "right"
            for paragraph in cell.findall("./w:p", NS):
                set_paragraph_alignment(paragraph, alignment)
                fixed += 1
    return fixed


def repair_document_xml(xml_path: Path) -> tuple[int, int, int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    fixed_ppr = merge_duplicate_paragraph_properties(root)
    fixed_jc = normalize_duplicate_alignments(root)
    fixed_sections = normalize_section_properties(root)
    fixed_bookmarks = normalize_body_bookmarks(root)
    fixed_ppr_order = normalize_paragraph_property_order(root)
    fixed_table = 0

    for table in root.findall(".//w:tbl", NS):
        if is_checklist_table(table):
            fixed_table += normalize_checklist_table(table)

    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
    return (
        fixed_ppr,
        fixed_jc,
        fixed_sections,
        fixed_bookmarks,
        fixed_ppr_order,
        fixed_table,
    )


def remove_relationships(relationships_path: Path, relationship_types: set[str]) -> int:
    if not relationships_path.exists():
        return 0

    tree = ET.parse(relationships_path)
    root = tree.getroot()
    removed = 0
    for relationship in list(root):
        if relationship.get("Type") in relationship_types:
            root.remove(relationship)
            removed += 1

    if removed:
        tree.write(relationships_path, encoding="UTF-8", xml_declaration=True)
    return removed


def remove_content_type_overrides(content_types_path: Path, part_names: set[str]) -> int:
    tree = ET.parse(content_types_path)
    root = tree.getroot()
    removed = 0
    for override in list(root):
        if override.tag == CT + "Override" and override.get("PartName") in part_names:
            root.remove(override)
            removed += 1

    if removed:
        tree.write(content_types_path, encoding="UTF-8", xml_declaration=True)
    return removed


def strip_unused_notes_and_comments(docx_root: Path) -> int:
    document_xml = (docx_root / "word" / "document.xml").read_text(encoding="utf-8")
    unused_parts: set[str] = set()
    relationship_types: set[str] = set()
    content_type_parts: set[str] = set()

    if "footnoteReference" not in document_xml:
        unused_parts.update(
            {
                "word/footnotes.xml",
                "word/_rels/footnotes.xml.rels",
            }
        )
        relationship_types.add(
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes"
        )
        content_type_parts.add("/word/footnotes.xml")

    if (
        "commentReference" not in document_xml
        and "commentRangeStart" not in document_xml
        and "commentRangeEnd" not in document_xml
    ):
        unused_parts.add("word/comments.xml")
        relationship_types.add(
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
        )
        content_type_parts.add("/word/comments.xml")

    removed = 0
    for relative_path in unused_parts:
        path = docx_root / relative_path
        if path.exists():
            path.unlink()
            removed += 1

    removed += remove_relationships(
        docx_root / "word" / "_rels" / "document.xml.rels",
        relationship_types,
    )
    removed += remove_content_type_overrides(
        docx_root / "[Content_Types].xml",
        content_type_parts,
    )
    return removed


def repair_docx(docx_path: Path) -> tuple[int, int, int, int, int, int, int]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with ZipFile(docx_path, "r") as zin:
            zin.extractall(temp_path)

        fixed = repair_document_xml(temp_path / "word" / "document.xml")
        stripped = strip_unused_notes_and_comments(temp_path)

        backup_path = docx_path.with_suffix(docx_path.suffix + ".tmp")
        with ZipFile(backup_path, "w", ZIP_DEFLATED) as zout:
            for path in sorted(temp_path.rglob("*")):
                if path.is_file():
                    zout.write(path, path.relative_to(temp_path).as_posix())

        shutil.move(backup_path, docx_path)
        return (*fixed, stripped)


def output_docx_paths() -> list[Path]:
    outputs = os.environ.get("QUARTO_PROJECT_OUTPUT_FILES", "")
    paths = [Path(line) for line in outputs.splitlines() if line.strip().endswith(".docx")]
    if paths:
        return paths

    fallback = Path("manuscript.docx")
    return [fallback] if fallback.exists() else []


def main() -> None:
    for docx_path in output_docx_paths():
        (
            fixed_ppr,
            fixed_jc,
            fixed_sections,
            fixed_bookmarks,
            fixed_ppr_order,
            fixed_table,
            stripped,
        ) = repair_docx(docx_path)
        print(
            "Repaired DOCX XML in "
            f"{docx_path}: merged {fixed_ppr} duplicate pPr block(s), "
            f"removed {fixed_jc} duplicate alignment block(s), "
            f"completed {fixed_sections} section property item(s), "
            f"moved {fixed_bookmarks} body-level bookmark(s), "
            f"reordered {fixed_ppr_order} paragraph property block(s), "
            f"aligned {fixed_table} checklist-table paragraph(s), "
            f"removed {stripped} unused notes/comment package item(s)."
        )


if __name__ == "__main__":
    main()
