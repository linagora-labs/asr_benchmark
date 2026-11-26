// Shared by the leaderboard and the contextual biasing page (put in their scripts when they are built)

// Each table's horizontal scrollbar sticks to the bottom of the window while the table is on
// screen (wrapper: the sticky bar stays within the table's extent). Drawn by the page (track +
// thumb to drag, click on the track to jump, wheel to scroll): the system's overlay scrollbars
// are too thin to catch.
function stickyScrollbar(box) {
  const wrap = box.ownerDocument.createElement("div"), bar = box.ownerDocument.createElement("div");
  wrap.className = "scroll-wrap";
  bar.className = "sticky-scrollbar";
  bar.innerHTML = `<div class="track"><div class="thumb"></div></div>`;
  box.replaceWith(wrap);
  wrap.append(box, bar);
  const track = bar.firstChild, thumb = track.firstChild;
  const range = () => box.scrollWidth - box.clientWidth;  // scrollable distance of the table
  const free = () => track.clientWidth - thumb.offsetWidth;  // movable distance of the thumb
  const update = () => {
    bar.hidden = range() <= 0;
    if (bar.hidden) return;
    thumb.style.width = `${100 * box.clientWidth / box.scrollWidth}%`;
    thumb.style.left = `${free() * box.scrollLeft / range()}px`;
  };
  box.addEventListener("scroll", update);
  thumb.onpointerdown = e => {
    e.preventDefault();
    e.stopPropagation();
    thumb.setPointerCapture(e.pointerId);
    thumb.classList.add("dragging");
    const x0 = e.clientX, left0 = box.scrollLeft;
    thumb.onpointermove = m => { box.scrollLeft = left0 + (m.clientX - x0) * range() / Math.max(free(), 1); };
    thumb.onpointerup = thumb.onpointercancel = () => {
      thumb.classList.remove("dragging");
      thumb.onpointermove = thumb.onpointerup = thumb.onpointercancel = null;
    };
  };
  // Click on the track: the thumb's center goes there
  track.onpointerdown = e => {
    if (e.target !== track) return;
    const x = e.clientX - track.getBoundingClientRect().left - thumb.offsetWidth / 2;
    box.scrollLeft = Math.max(0, Math.min(1, x / Math.max(free(), 1))) * range();
  };
  bar.addEventListener("wheel", e => {
    box.scrollLeft += e.deltaX || e.deltaY;
    e.preventDefault();
  }, { passive: false });
  const observer = new ResizeObserver(update);  // window resize, tab shown, table re-rendered
  observer.observe(box);
  observer.observe(box.querySelector("table"));
}

// The header row of each table follows the window's scroll while the table is on screen (its
// .scroll box scrolls horizontally, so position: sticky would stick within the box, not the window):
// moved down by the distance scrolled past the table's top, until the table's last row.
function stickyHeaders(tables) {
  tables.forEach(table => {
    const head = table.tHead;
    if (!head) return;
    const top = table.getBoundingClientRect().top, max = table.offsetHeight - head.offsetHeight;
    const shift = Math.max(0, Math.min(-top, max));
    head.style.transform = shift ? `translateY(${shift}px)` : "";
    head.classList.toggle("stuck", shift > 0);
  });
}

// Both for the tables (.scroll) of a page
function setUpTables(root) {
  root.querySelectorAll(".scroll").forEach(box => box.querySelector("table") && stickyScrollbar(box));
  const tables = [...root.querySelectorAll(".scroll table")], update = () => stickyHeaders(tables);
  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
  // Tables re-rendered (sort, filters, tab): their new header goes back in place
  tables.forEach(table => new MutationObserver(update).observe(table, { childList: true }));
}

// Model info
const params = n => n >= 1e9 ? `${(n / 1e9).toFixed(n < 1e10 ? 1 : 0)}B` : `${Math.round(n / 1e6)}M`;
const licenseText = l => ({ "apache-2.0": "Apache-2.0", gemma: "Gemma" })[l] || l.toUpperCase();
// Official text of the licenses, linked from their name
const LICENSE_URLS = {
  "apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
  mit: "https://opensource.org/license/mit",
  "cc-by-4.0": "https://creativecommons.org/licenses/by/4.0/",
  "cc-by-nc-4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
  "openmdw-1.1": "https://openmdw.ai/license/1-1/",
  gemma: "https://ai.google.dev/gemma/terms",
};
const licenseHtml = l => {
  const text = String(licenseText(l)).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  return LICENSE_URLS[l] ? `<a href="${LICENSE_URLS[l]}" target="_blank" rel="noopener">${text}</a>` : text;
};
