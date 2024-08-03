window.MathJax = {
  loader: {load: ['[tex]/physics']},
  tex: {
    packages: {'[+]': ['physics']},
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    macros: {
      coloneqq: "\\mathrel{:=}",
      eqqcolon: "\\mathrel{=:}",
      divg: "\\grad\\vdot",
      Vec: ["\\vec{\\boldsymbol{#1}}", 1],
      VecVec: ["\\widetilde{\\boldsymbol{#1}}", 1],
      Mat: ["\\mathinner{\\underline{#1}}", 1],
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
