-- Markdown uses editable SVGs; LaTeX embeds their matching vector PDFs.
function Image(image)
  if FORMAT:match("latex") then
    image.src = image.src:gsub("%.svg$", ".pdf")
  end
  return image
end

function Table(tbl)
  if #tbl.colspecs == 3 and
      pandoc.utils.stringify(tbl.head.rows[1].cells[1].contents) == "Order" then
    local widths = {0.23, 0.18, 0.59}
    for i, spec in ipairs(tbl.colspecs) do
      spec[2] = widths[i]
    end
  end
  if #tbl.colspecs == 6 and
      pandoc.utils.stringify(tbl.head.rows[1].cells[1].contents) == "Dimension" then
    local widths = {0.18, 0.08, 0.09, 0.10, 0.10, 0.45}
    for i, spec in ipairs(tbl.colspecs) do
      spec[2] = widths[i]
    end
  end
  -- Both seven-column tables need room for the descriptive first column.
  if #tbl.colspecs == 7 then
    for i, spec in ipairs(tbl.colspecs) do
      spec[2] = i == 1 and 0.22 or 0.13
    end
  end
  if FORMAT:match("latex") then
    -- These compact tables each fit on one page; keep comparisons together.
    return {
      pandoc.RawBlock("latex", "\\begin{minipage}{\\linewidth}"),
      tbl,
      pandoc.RawBlock("latex", "\\end{minipage}")
    }
  end
  return tbl
end

function Header(header)
  if FORMAT:match("latex") then
    local title = pandoc.utils.stringify(header.content)
    if title:match("^Appendix [ABC]%.") then
      return {pandoc.RawBlock("latex", "\\clearpage"), header}
    elseif title == "13. Acceptance outlook and the revisions needed today" then
      return {pandoc.RawBlock("latex", "\\Needspace{12\\baselineskip}"), header}
    elseif title == "References" then
      return {pandoc.RawBlock("latex", "\\Needspace{18\\baselineskip}"), header}
    elseif title:match("^%d+%. Conclusion$") then
      return {pandoc.RawBlock("latex", "\\Needspace{14\\baselineskip}"), header}
    end
  end
  return header
end
