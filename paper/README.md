# BD-SHS Hate Speech Classification Paper

This directory contains the IEEE-style LaTeX paper for the hierarchical aggregation attribution study.

## Files

- `paper.tex` - Main LaTeX document
- `references.bib` - Bibliography file
- `README.md` - This file

## Overleaf Instructions

1. Go to [Overleaf](https://www.overleaf.com/)
2. Create a new project
3. Upload `paper.tex` and `references.bib`
4. Set the compiler to XeLaTeX or LuaLaTeX (recommended for better Unicode support)
5. Click "Recompile" to generate the PDF

## Local Compilation

### Prerequisites
- TeX Live or MiKTeX
- LaTeX editor (TeXShop, TeXworks, VS Code with LaTeX Workshop, etc.)

### Compilation Commands

```bash
# Using pdflatex (standard)
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex

# Using xelatex (recommended for Unicode)
xelatex paper.tex
bibtex paper
xelatex paper.tex
xelatex paper.tex

# Using latexmk (automates the process)
latexmk -pdf paper.tex
```

## Paper Structure

### Sections
1. Introduction
2. Background
3. Methodology
4. Results
5. Discussion
6. Conclusion

### Tables
- Table 1: Model Performance Metrics
- Table 2: Tokenization Statistics
- Table 3: Main Faithfulness Results
- Table 4: Statistical Comparison Results
- Table 5: Coverage Analysis

### Key Results
- Hierarchical aggregation improves sufficiency efficiency with medium-large effect sizes (d_z = -0.38 to -0.59)
- 33/36 comparisons significant after Holm-Bonferroni correction
- Coverage discrepancy: Hierarchical achieves 85% higher actual word coverage at 10% budget

## Limitations Documented in Paper

1. **Coverage Discrepancy**: Hierarchical achieves higher actual word coverage than flat strategies
2. **Single Seed**: Evaluation uses only seed 42 (multi-seed testing deferred)
3. **SHAP Evaluation**: Main analysis focuses on Integrated Gradients (SHAP deferred)
4. **Qualitative Analysis**: Limited to efficiency metrics (manual labeling deferred)
5. **Model-Specific Results**: Combined evaluation (model-specific deferred)

## References

The paper cites:
- Integrated Gradients (Sundararajan et al., 2017)
- SHAP (Lundberg & Lee, 2017)
- Byte-Pair Encoding (Sennrich et al., 2015)
- WordPiece (Schuster & Nakajima, 2012)
- BD-SHS Dataset (Anonymous, 2023)
- BanglaBERT (Islam et al., 2020)
- XLM-R (Conneau et al., 2019)

## Notes

- The paper uses IEEEtran document class with journal style
- All tables use booktabs for professional formatting
- The paper includes comprehensive statistical validation (bootstrap CI, permutation tests, effect sizes, Holm-Bonferroni correction)
- Test metrics in Table 1 are marked as N/A due to ground truth matching requirements
- Coverage analysis includes estimated budgets for equal-coverage comparison
