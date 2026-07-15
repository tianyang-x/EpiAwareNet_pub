# EpiAwareNet — Research Poster

A modern, single-page A0 (landscape) conference poster summarizing EpiAwareNet:
an epigenome-aware Transformer for gene regulatory network (GRN) inference from
single-cell multi-omics data.

## Files

| File | Description |
| --- | --- |
| `epiawarenet_poster.tex` | Poster source (`beamerposter` + `tcolorbox`). |
| `epiawarenet_poster.pdf` | Compiled poster (A0 landscape, one page). |

## Building

The poster uses `beamer`/`beamerposter`, `tikz`, `tcolorbox`, and `fontawesome5`,
all part of a standard TeX Live install.

```bash
# Option A: pdflatex (run once; no bibliography needed)
pdflatex epiawarenet_poster.tex

# Option B: tectonic (self-contained, fetches packages on demand)
tectonic epiawarenet_poster.tex
```

Required TeX Live packages: `texlive-latex-recommended`, `texlive-latex-extra`
(provides `beamerposter`, `tcolorbox`), `texlive-pictures` (TikZ),
`texlive-fonts-extra` (`fontawesome5`), and `lmodern`.

## Layout

- **Header** — title, subtitle, author, repository, KDD badge.
- **Column 1** — Motivation, Key Contributions, At a Glance.
- **Column 2** — Model Architecture (dual-path backbone diagram),
  Two-Stage Learning (masked NB + nnPU losses), Design Notes & Scalability.
- **Column 3** — Context-Specific GRNs, Evaluation Protocol, Ablation Studies,
  Takeaways.
- **Bottom band** — end-to-end pipeline flow.

## Customizing

- **Colours:** edit the `\definecolor` block near the top of the `.tex`.
- **Physical size:** change `size=a0` / `orientation` in the `beamerposter`
  options; adjust `scale` to trade content density for font size.
- **Author / affiliation:** edit the header banner minipages.
