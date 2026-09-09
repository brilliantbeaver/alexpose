-- Keep the readable Markdown as the source, using vector PDF assets in print.
function Code(code)
  if FORMAT:match('latex') then
    local rendered = pandoc.write(pandoc.Pandoc({pandoc.Plain({code})}), 'latex')
    rendered = rendered:gsub('\\_', '\\_\\allowbreak{}')
    rendered = rendered:gsub('/', '/\\allowbreak{}')
    return pandoc.RawInline('latex', rendered:gsub('%s+$', ''))
  end
  return code
end

function Image(image)
  if image.src:match('%.svg$') then
    image.src = image.src:gsub('%.svg$', '.pdf')
    image.attributes.width = '100%'
  end
  return image
end

-- Explicit widths prevent prose columns from overflowing the handout margins.
function Table(tbl)
  local widths
  if #tbl.colspecs == 2 then
    widths = {0.37, 0.63}
  elseif #tbl.colspecs == 3 then
    if pandoc.utils.stringify(tbl.head):match('Horizon') then
      widths = {0.24, 0.38, 0.38}
    else
      widths = {0.65, 0.12, 0.23}
    end
  elseif #tbl.colspecs == 5 then
    if pandoc.utils.stringify(tbl.head):match('Workshop or track') then
      widths = {0.20, 0.08, 0.10, 0.20, 0.42}
    else
      widths = {0.38, 0.16, 0.14, 0.14, 0.18}
    end
  else
    return tbl
  end
  for index, width in ipairs(widths) do
    tbl.colspecs[index][2] = width
  end
  return tbl
end
