function copyBibTeX() {
  const code = document.getElementById("bibtex-code");
  if (!code) return;
  const text = code.innerText;
  navigator.clipboard.writeText(text).then(() => {
    const button = document.querySelector(".paperpage-copy-btn span:last-child");
    if (!button) return;
    const original = button.textContent;
    button.textContent = "Copied";
    window.setTimeout(() => {
      button.textContent = original;
    }, 1200);
  });
}

function fitHeroTitle() {
  const title = document.querySelector(".publication-title");
  if (!title) return;

  const desktop = window.matchMedia("(min-width: 1100px)").matches;
  title.style.fontSize = "";

  if (!desktop) return;

  const computed = window.getComputedStyle(title);
  const lineHeight = parseFloat(computed.lineHeight);
  let fontSize = parseFloat(computed.fontSize);
  const minFontSize = 42;
  const maxHeight = lineHeight * 2 + 2;

  while (title.offsetHeight > maxHeight && fontSize > minFontSize) {
    fontSize -= 1;
    title.style.fontSize = `${fontSize}px`;
  }
}

window.addEventListener("DOMContentLoaded", fitHeroTitle);
window.addEventListener("resize", () => {
  window.clearTimeout(window.__paperpageTitleFitTimer);
  window.__paperpageTitleFitTimer = window.setTimeout(fitHeroTitle, 80);
});
