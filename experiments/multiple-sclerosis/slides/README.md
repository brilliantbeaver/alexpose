# Slides

`slides.md` is a [Marp](https://marp.app) deck summarizing the whole S-JEPA gait study,
step by step, with the same vector diagrams the notebooks use (from `../images`).

## View or export

The easiest way is the **Marp for VS Code** extension: open `slides.md` and click the
preview icon.

From the command line (needs Node.js):

```bash
# HTML (self-contained, opens in any browser)
npx @marp-team/marp-cli@latest slides.md -o slides.html

# PDF
npx @marp-team/marp-cli@latest slides.md --pdf

# PNG per slide
npx @marp-team/marp-cli@latest slides.md --images png
```

GitHub also renders `slides.md` as plain markdown, so the content is readable without any
tooling.
