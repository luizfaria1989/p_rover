// Desenha as fórmulas escritas com $...$ (no texto) e $$...$$ (em bloco).
// O document$ do Material dispara a cada troca de página (navigation.instant).
document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
  });
});
