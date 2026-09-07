-- Preserve editable SVGs in Markdown, use vector PDFs in the manuscript.
function Image(el)
  if FORMAT:match("latex") then
    el.src = el.src:gsub("%.svg$", ".pdf")
  end
  return el
end

function Header(el)
  if FORMAT:match("latex") and pandoc.utils.stringify(el.content):match("^Appendix A%.") then
    return {pandoc.RawBlock("latex", "\\FloatBarrier\\clearpage"), el}
  end
end

function Table(el)
  local label = pandoc.utils.stringify(el.head.rows[1].cells[1].contents)
  local widths
  if label == "Design" then
    widths = {0.18, 0.40, 0.42}
  elseif label == "Study A readout" then
    widths = {0.52, 0.23, 0.25}
  elseif label == "Claim" then
    widths = {0.26, 0.40, 0.34}
  elseif label == "Contrast" then
    widths = {0.46, 0.12, 0.16, 0.26}
  elseif label == "Lane" then
    widths = {0.42, 0.20, 0.20, 0.18}
  end
  if widths then
    for i, width in ipairs(widths) do
      el.colspecs[i][2] = width
    end
  end
  return el
end
