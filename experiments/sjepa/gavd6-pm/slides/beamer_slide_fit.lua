-- Let Beamer scale a frame down when its content would otherwise overflow the page.
--
-- Beamer's "shrink" option only acts when a frame actually overflows, so marking every
-- slide is safe: frames that already fit are left at full size. This runs as a filter on
-- the PDF build rather than as an attribute in the Markdown, so the same source keeps
-- producing clean HTML and PowerPoint.

function Header(element)
  if element.level == 1 then
    element.classes:insert("shrink")
  end
  return element
end
