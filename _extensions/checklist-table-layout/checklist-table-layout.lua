local target_table_id = "tbl-checklist-grouped"
local target_headers = {
  "Checklist item",
  "TP",
  "FP",
  "FN",
  "TN",
  "Precision",
  "Recall",
  "F1",
}

local function trim(text)
  return text:gsub("^%s+", ""):gsub("%s+$", "")
end

local function cell_text(cell)
  return trim(pandoc.utils.stringify(cell.contents))
end

local function has_target_headers(tbl)
  if #tbl.colspecs ~= #target_headers then
    return false
  end

  local header_row = tbl.head.rows[1]
  if header_row == nil or #header_row.cells ~= #target_headers then
    return false
  end

  for index, expected in ipairs(target_headers) do
    if cell_text(header_row.cells[index]) ~= expected then
      return false
    end
  end

  return true
end

local function column_alignment(index)
  if index == 1 then
    return pandoc.AlignLeft
  end
  return pandoc.AlignRight
end

local function align_colspecs(colspecs)
  for index, spec in ipairs(colspecs) do
    colspecs[index] = { column_alignment(index), spec[2] }
  end
end

local function row_is_category(row)
  if #row.cells == 0 or cell_text(row.cells[1]) == "" then
    return false
  end

  for index = 2, #row.cells do
    if cell_text(row.cells[index]) ~= "" then
      return false
    end
  end

  return true
end

local function plain_cell_blocks(text, bold)
  local inlines = pandoc.Inlines({})
  if text ~= "" then
    if bold then
      inlines:insert(pandoc.Strong({ pandoc.Str(text) }))
    else
      inlines:insert(pandoc.Str(text))
    end
  end
  return pandoc.Blocks({ pandoc.Plain(inlines) })
end

local function align_row(row, is_header)
  local is_category = row_is_category(row)

  for index, cell in ipairs(row.cells) do
    local text = cell_text(cell)
    local bold = is_header or (is_category and index == 1)
    cell.alignment = column_alignment(index)
    cell.contents = plain_cell_blocks(text, bold)
    row.cells[index] = cell
  end
end

local function align_rows(rows, is_header)
  if rows == nil then
    return
  end

  for index, row in ipairs(rows) do
    align_row(row, is_header)
    rows[index] = row
  end
end

function Table(tbl)
  if tbl.identifier ~= target_table_id and not has_target_headers(tbl) then
    return nil
  end

  align_colspecs(tbl.colspecs)
  align_rows(tbl.head.rows, true)

  for _, body in ipairs(tbl.bodies) do
    align_rows(body.head, false)
    align_rows(body.body, false)
  end

  align_rows(tbl.foot.rows, false)
  return tbl
end
