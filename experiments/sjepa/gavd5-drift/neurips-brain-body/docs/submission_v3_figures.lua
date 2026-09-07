-- V3-only layout; no changes to V2 or its shared figure assets.
function Image(el)
  if FORMAT:match("latex") then
    el.src = el.src:gsub("%.svg$", ".pdf")
  end
  return el
end

-- Keep the one-page pipeline overview below its appendix heading and text.
function Figure(el)
  if FORMAT:match("latex") then
    local pipeline = false
    el:walk({Image = function(img)
      if img.src:match("submission_pipeline_v3%.") then pipeline = true end
    end})
    if pipeline then
      local latex = pandoc.write(pandoc.Pandoc({el}), "latex")
      latex = latex:gsub("\\begin{figure}", "\\begin{figure}[H]", 1)
      return pandoc.RawBlock("latex", latex)
    end
  end
end

function Header(el)
  if FORMAT:match("latex") then
    local title = pandoc.utils.stringify(el.content)
    if title:match("^Appendix [ABC]%.") then
      return {pandoc.RawBlock("latex", "\\FloatBarrier\\clearpage"), el}
    elseif title:match("^A%.4 Checks") then
      return {pandoc.RawBlock("latex", "\\FloatBarrier\\Needspace{8\\baselineskip}"), el}
    elseif title:match("^A%.5 Reading Figure") then
      return {pandoc.RawBlock("latex", "\\Needspace{8\\baselineskip}"), el}
    elseif title:match("^5%. Implications") then
      return {pandoc.RawBlock("latex", "\\Needspace{10\\baselineskip}"), el}
    end
  end
end

-- Keep the feature-disagreement explanation with its equation.
function Para(el)
  if FORMAT:match("latex") then
    local text = pandoc.utils.stringify(el.content)
    if text:match("^Feature agreement is measured directly") then
      return {pandoc.RawBlock("latex", "\\Needspace{13\\baselineskip}"), el}
    end
  end
end

function Table(el)
  local label = pandoc.utils.stringify(el.head.rows[1].cells[1].contents)
  local widths
  if label == "Question" then
    widths = {0.43, 0.29, 0.28}
  elseif label == "Features used" then
    widths = {0.46, 0.29, 0.25}
  elseif label == "Method" then
    widths = {0.43, 0.17, 0.17, 0.23}
  end
  if widths then
    for i, width in ipairs(widths) do el.colspecs[i][2] = width end
  end
  return el
end
