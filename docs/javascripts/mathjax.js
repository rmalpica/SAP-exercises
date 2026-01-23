// MathJax configuration (optional but useful)
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  },
  options: {
    // Process only elements that contain math (keeps it fast)
    ignoreHtmlClass: '.*',
    processHtmlClass: 'arithmatex'
  }
};

// Re-render MathJax on every MkDocs Material page navigation
document$.subscribe(() => {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise();
  }
});

