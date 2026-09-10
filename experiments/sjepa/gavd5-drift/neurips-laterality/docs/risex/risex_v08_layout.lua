-- Keep scientific content in the Markdown; only alter its LaTeX presentation.
local function latex(blocks)
  return pandoc.write(pandoc.Pandoc(blocks), "latex"):gsub("%s+$", "")
end

function Header(h)
  return pandoc.RawBlock("latex", "\\risexheading{" .. latex({pandoc.Plain(h.content)}) .. "}")
end

function Para(p)
  if pandoc.utils.stringify(p):match("^Table 1%.") then
    return pandoc.RawBlock("latex", "\\begin{minipage}{\\linewidth}\n{\\fontsize{9}{10.5}\\selectfont " .. latex({p}) .. "\\par}\n\\vspace{3pt}")
  end
end

function Table(t)
  local lines = {
    "\\setlength{\\tabcolsep}{2pt}",
    "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}Xrr@{}}",
    "\\toprule"
  }
  local function row(r)
    local cells = {}
    for _, c in ipairs(r.cells) do
      cells[#cells + 1] = latex(c.contents)
    end
    lines[#lines + 1] = table.concat(cells, " & ") .. " \\\\"
  end
  for _, r in ipairs(t.head.rows) do row(r) end
  lines[#lines + 1] = "\\midrule"
  for _, b in ipairs(t.bodies) do
    for _, r in ipairs(b.head) do row(r) end
    for _, r in ipairs(b.body) do row(r) end
  end
  lines[#lines + 1] = "\\bottomrule\n\\end{tabularx}\n\\end{minipage}"
  return pandoc.RawBlock("latex", table.concat(lines, "\n"))
end
