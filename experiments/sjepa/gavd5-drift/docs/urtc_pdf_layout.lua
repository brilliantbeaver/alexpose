-- Keep two-column explanatory tables inside the printable page width.
-- Pandoc otherwise emits natural-width LaTeX columns, which can overflow
-- when a cell contains a long technical phrase.
function Table(table_element)
  if #table_element.colspecs == 2 then
    table_element.colspecs[1][2] = 0.30
    table_element.colspecs[2][2] = 0.70
  elseif #table_element.colspecs == 3 then
    table_element.colspecs[1][2] = 0.17
    table_element.colspecs[2][2] = 0.28
    table_element.colspecs[3][2] = 0.55
  end

  return table_element
end

-- Keep PDF-only page breaks invisible in the readable Markdown. The source
-- uses an empty HTML div, which browsers ignore, and this filter translates
-- that marker into a LaTeX page break during the PDF build.
function RawBlock(block)
  if block.format == "html" and
      string.find(block.text, "pdf-page-break", 1, true) then
    return pandoc.RawBlock("latex", "\\clearpage")
  end

  return block
end

function Div(block)
  for _, class_name in ipairs(block.classes) do
    if class_name == "pdf-page-break" then
      return pandoc.RawBlock("latex", "\\clearpage")
    end
  end

  return block
end

-- The Markdown already has a compact web ToC. For PDF, remove that block so
-- Pandoc can create a native, page-numbered, clickable ToC instead. Also avoid
-- repeating the document title as both a title block and a level-one heading:
-- any level-1 heading whose text matches the --metadata title passed on the
-- command line is dropped, since the PDF title block already renders it.
-- Smart typography turns a straight apostrophe in the Markdown heading into a
-- curly one, but a title passed via -M on the command line is not reprocessed
-- that way; normalize both sides so the two still compare equal.
local function normalize_quotes(text)
  text = text:gsub("\226\128\152", "'") -- U+2018 LEFT SINGLE QUOTATION MARK
  text = text:gsub("\226\128\153", "'") -- U+2019 RIGHT SINGLE QUOTATION MARK
  return text
end

function Pandoc(document)
  local blocks = {}
  local skipping_web_toc = false
  local metadata_title = ""
  if document.meta.title then
    metadata_title = normalize_quotes(pandoc.utils.stringify(document.meta.title))
  end

  for _, block in ipairs(document.blocks) do
    local heading = ""
    if block.t == "Header" then
      heading = normalize_quotes(pandoc.utils.stringify(block.content))
    end

    if block.t == "Header" and block.level == 1 and metadata_title ~= "" and
        heading == metadata_title then
      -- The PDF metadata renders this title separately.
    elseif block.t == "Header" and heading == "Table of contents" then
      skipping_web_toc = true
    elseif skipping_web_toc and block.t == "Header" and
        heading == "How to use this tutorial" then
      skipping_web_toc = false
      table.insert(blocks, block)
    elseif not skipping_web_toc then
      table.insert(blocks, block)
    end
  end

  document.blocks = blocks
  return document
end
