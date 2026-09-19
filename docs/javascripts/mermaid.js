document$.subscribe(() => {
  if (window.mermaid) {
    window.mermaid.initialize({ startOnLoad: false });
    window.mermaid.run({ querySelector: ".mermaid" });
  }
});
