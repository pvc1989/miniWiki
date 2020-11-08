window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    macros: {
      coloneqq: "\\mathrel{:=}",
      eqqcolon: "\\mathrel{=:}",
      cross: "\\boldsymbol{\\times}",
      vdot: "\\mathbin{\\boldsymbol{\\cdot}}",
      grad: "\\boldsymbol{\\nabla}",
      divg: "\\grad\\vdot",
      curl: "\\grad\\cross",
      abs: ["\\left\\vert{#1}\\right\\vert", 1],
      Vec: ["\\vec{\\boldsymbol{#1}}", 1],
      VecVec: ["\\widetilde{\\boldsymbol{#1}}", 1],
      ket: ["\\mathinner{\\vert#1\\rangle}", 1],
      Mat: ["\\mathinner{\\underline{#1}}", 1],
      ip: ["\\mathinner{\\langle#1\\vert#2\\rangle}", 2],
      dd: ["\\mathinner{\\mathopen{\\mathrm{d}}#1}", 1],
      dv: ["\\frac{\\mathopen{\\mathrm{d}}#1}{\\mathopen{\\mathrm{d}}#2}", 2],
      pdv: ["\\frac{\\mathopen{\\partial}#1}{\\mathopen{\\partial}#2}", 2],
    },
  },
};
(function () {
  function stripCDATA(x) {
    if (x.startsWith('% <![CDATA[') && x.endsWith('%]]>'))
      return x.substring(11,x.length-4);
    return x;
  }
  document.querySelectorAll("script[type='math/tex']").forEach(function(el) {
    el.outerHTML = "\\(" + stripCDATA(el.textContent) + "\\)";
  });
  document.querySelectorAll("script[type='math/tex; mode=display']").forEach(function(el) {
    el.outerHTML = "\\[" + stripCDATA(el.textContent) + "\\]";
  });
  var script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js';
  script.async = true;
  document.head.appendChild(script);
})();
