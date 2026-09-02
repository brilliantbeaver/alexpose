-- Keep two-column explanatory tables inside the printable page width.
-- Pandoc otherwise emits natural-width LaTeX columns, which can overflow
-- when a cell contains a long technical phrase.
function Table(table_element)
  if #table_element.colspecs == 2 then
    table_element.colspecs[1][2] = 0.30
    table_element.colspecs[2][2] = 0.70
  end

  return table_element
end

-- The Markdown already has a compact web ToC. For PDF, remove that block so
-- Pandoc can create a native, page-numbered, clickable ToC instead. Also avoid
-- repeating the document title as both a title block and a level-one heading.
function Pandoc(document)
  local blocks = {}
  local skipping_web_toc = false
  local tutorial_title =
    "A Step-by-Step Tutorial for the Entire URTC S-JEPA Gait Paper"

  for _, block in ipairs(document.blocks) do
    local heading = ""
    if block.t == "Header" then
      heading = pandoc.utils.stringify(block.content)
    end

    if block.t == "Header" and block.level == 1 and heading == tutorial_title then
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
