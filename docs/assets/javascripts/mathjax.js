// SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
//
// SPDX-License-Identifier: MIT

window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"], ["$", "$"]],
      displayMath: [["\\[", "\\]"], ["$$", "$$"]],
      processEscapes: true,
      processEnvironments: true
    }
  };

  document$.subscribe(() => {


    MathJax.startup.output.clearCache()
    MathJax.typesetClear()
    MathJax.texReset()
    MathJax.typesetPromise()
  })
