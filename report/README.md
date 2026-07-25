# Final report source

`main.tex` is the canonical source for the Korean five-page competition report.
The document is A4, uses a compact two-column academic layout, and keeps all
non-graph elements monochrome.

Build it with Tectonic 0.16.9 and Poppler:

```bash
python -B scripts/build_report.py \
  --tectonic /path/to/tectonic \
  --pdfinfo /path/to/pdfinfo
```

The build fails on an overfull box, a non-A4 page, or a page count other than
five. The generated release document replaces
`docs/final_report_5page_ko.pdf`.

The numerical contract is `docs/final_results.json`. Final-model Public
ablations and historical clean954 development diagnostics are intentionally
separated in the report; cross-panel rows are not causal comparisons.
