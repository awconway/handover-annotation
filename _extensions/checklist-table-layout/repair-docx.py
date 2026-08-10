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

TABLE_LAYOUTS = [
    {
        "headers": ["Category", "Checklist item"],
        "alignments": ["left", "left"],
        "widths": [30, 70],
    },
    {
        "headers": ["Category", "Definition / when to use", "Example from handover speech"],
        "alignments": ["left", "left", "left"],
        "widths": [22, 43, 35],
    },
    {
        "headers": [
            "Label",
            "Gold",
            "Predicted spans",
            "Recall",
            "Precision",
            "Mean IoU",
            "F1",
        ],
        "alignments": ["left", "right", "right", "right", "right", "right", "right"],
        "widths": [18, 8, 13, 14, 14, 15, 18],
    },
    {
        "headers": ["Checklist item", "Accuracy", "Precision", "Recall", "F1"],
        "alignments": ["left", "right", "right", "right", "right"],
        "widths": [44, 14, 14, 14, 14],
    },
    {
        "headers": ["Task", "Model", "Approach", "Precision", "Recall", "F1", "Mean IoU"],
        "alignments": ["left", "left", "left", "right", "right", "right", "right"],
        "widths": [16, 15, 15, 13, 13, 13, 15],
    },
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


def set_paragraph_single_spacing(paragraph: ET.Element) -> None:
    ppr = paragraph_properties(paragraph)
    for spacing in direct_children(ppr, W + "spacing"):
        ppr.remove(spacing)

    spacing = ET.Element(W + "spacing")
    spacing.set(W + "before", "0")
    spacing.set(W + "after", "0")
    spacing.set(W + "line", "240")
    spacing.set(W + "lineRule", "auto")
    ppr.append(spacing)


def cell_properties(cell: ET.Element) -> ET.Element:
    tcprs = direct_children(cell, W + "tcPr")
    if tcprs:
        return tcprs[0]

    tcpr = ET.Element(W + "tcPr")
    cell.insert(0, tcpr)
    return tcpr


def set_cell_width(cell: ET.Element, width: int) -> None:
    tcpr = cell_properties(cell)
    for tcw in direct_children(tcpr, W + "tcW"):
        tcpr.remove(tcw)

    tcw = ET.Element(W + "tcW")
    tcw.set(W + "w", str(width))
    tcw.set(W + "type", "dxa")
    tcpr.insert(0, tcw)


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


def collapse_trailing_landscape_section(root: ET.Element) -> int:
    """Avoid a blank portrait page after a landscape section at document end."""
    body = root.find("./w:body", NS)
    if body is None:
        return 0

    children = list(body)
    if len(children) < 2 or children[-1].tag != W + "sectPr":
        return 0

    trailing_paragraph = children[-2]
    if trailing_paragraph.tag != W + "p" or text_content(trailing_paragraph):
        return 0

    paragraph_sections = trailing_paragraph.findall("./w:pPr/w:sectPr", NS)
    if len(paragraph_sections) != 1:
        return 0

    section = paragraph_sections[0]
    page_size = section.find("./w:pgSz", NS)
    if page_size is None or page_size.get(W + "orient") != "landscape":
        return 0

    paragraph_properties_node = trailing_paragraph.find("./w:pPr", NS)
    if paragraph_properties_node is None:
        return 0

    paragraph_properties_node.remove(section)
    body.remove(trailing_paragraph)
    body.remove(children[-1])
    body.append(section)
    return 1


def normalize_list_paragraph_style(root: ET.Element) -> int:
    fixed = 0
    for ppr in root.findall(".//w:pPr", NS):
        num_pr = direct_children(ppr, W + "numPr")
        if not num_pr:
            continue

        p_style = direct_children(ppr, W + "pStyle")
        if not p_style:
            continue

        style = p_style[0]
        if style.get(W + "val") == "Compact":
            style.set(W + "val", "ListBullet")
            fixed += 1

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


def table_header(table: ET.Element) -> list[str]:
    rows = table_rows(table)
    if not rows:
        return []
    return [text_content(cell) for cell in row_cells(rows[0])]


def table_layout_for(table: ET.Element) -> dict[str, list] | None:
    header = table_header(table)
    for layout in TABLE_LAYOUTS:
        if header == layout["headers"]:
            return layout
    return None


def ensure_table_grid(table: ET.Element) -> ET.Element:
    grids = direct_children(table, W + "tblGrid")
    if grids:
        return grids[0]

    grid = ET.Element(W + "tblGrid")
    tbl_prs = direct_children(table, W + "tblPr")
    insert_index = 1 if tbl_prs else 0
    table.insert(insert_index, grid)
    return grid


def normalize_table_grid(table: ET.Element, widths: list[int]) -> int:
    grid = ensure_table_grid(table)
    existing_cols = direct_children(grid, W + "gridCol")
    current_total = 0
    for col in existing_cols:
        try:
            current_total += int(col.get(W + "w") or "0")
        except ValueError:
            pass
    if current_total <= 0:
        current_total = 7920

    width_total = sum(widths)
    target_widths = [
        max(120, round(current_total * width / width_total))
        for width in widths
    ]

    changed = 0
    while len(existing_cols) < len(target_widths):
        col = ET.Element(W + "gridCol")
        grid.append(col)
        existing_cols.append(col)
        changed += 1
    while len(existing_cols) > len(target_widths):
        grid.remove(existing_cols.pop())
        changed += 1

    for col, target_width in zip(existing_cols, target_widths):
        target = str(target_width)
        if col.get(W + "w") != target:
            col.set(W + "w", target)
            changed += 1

    return changed


def target_grid_widths(table: ET.Element, widths: list[int]) -> list[int]:
    grid = ensure_table_grid(table)
    existing_cols = direct_children(grid, W + "gridCol")
    current_total = 0
    for col in existing_cols:
        try:
            current_total += int(col.get(W + "w") or "0")
        except ValueError:
            pass
    if current_total <= 0:
        current_total = 7920

    width_total = sum(widths)
    return [max(120, round(current_total * width / width_total)) for width in widths]


def normalize_table_layout(table: ET.Element, layout: dict[str, list]) -> int:
    fixed = 0
    fixed += normalize_table_grid(table, layout["widths"])
    target_widths = target_grid_widths(table, layout["widths"])

    for row in table_rows(table):
        for column_index, cell in enumerate(row_cells(row)):
            if column_index >= len(layout["alignments"]):
                continue
            alignment = layout["alignments"][column_index]
            set_cell_width(cell, target_widths[column_index])
            for paragraph in cell.findall("./w:p", NS):
                set_paragraph_alignment(paragraph, alignment)
                set_paragraph_single_spacing(paragraph)
                fixed += 1
    return fixed


def repair_document_xml(xml_path: Path) -> tuple[int, int, int, int, int, int, int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    fixed_ppr = merge_duplicate_paragraph_properties(root)
    fixed_jc = normalize_duplicate_alignments(root)
    fixed_sections = normalize_section_properties(root)
    fixed_bookmarks = normalize_body_bookmarks(root)
    fixed_sections += collapse_trailing_landscape_section(root)
    fixed_lists = normalize_list_paragraph_style(root)
    fixed_ppr_order = normalize_paragraph_property_order(root)
    fixed_table = 0

    for table in root.findall(".//w:tbl", NS):
        layout = table_layout_for(table)
        if layout is not None:
            fixed_table += normalize_table_layout(table, layout)

    fixed_ppr_order += normalize_paragraph_property_order(root)

    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
    return (
        fixed_ppr,
        fixed_jc,
        fixed_sections,
        fixed_bookmarks,
        fixed_lists,
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


def repair_docx(docx_path: Path) -> tuple[int, int, int, int, int, int, int, int]:
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
            fixed_lists,
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
            f"updated {fixed_lists} list paragraph style(s), "
            f"reordered {fixed_ppr_order} paragraph property block(s), "
            f"aligned {fixed_table} checklist-table paragraph(s), "
            f"removed {stripped} unused notes/comment package item(s)."
        )


if __name__ == "__main__":
    main()
