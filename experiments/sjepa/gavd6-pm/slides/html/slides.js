(() => {
  "use strict";

  const viewport = document.getElementById("viewport");
  const deck = document.getElementById("deck");
  const slides = Array.from(deck.querySelectorAll(":scope > section"));
  const counter = document.getElementById("slide-counter");
  const progressBar = document.getElementById("progress-bar");
  const liveStatus = document.getElementById("live-status");
  const notesPanel = document.getElementById("notes-panel");
  const notesContent = document.getElementById("notes-content");
  const overviewOverlay = document.getElementById("overview-overlay");
  const overviewGrid = document.getElementById("overview-grid");
  const helpOverlay = document.getElementById("help-overlay");
  const focusableSelector = "a[href], button, input, select, textarea, [tabindex]";

  let currentIndex = 0;
  let touchStart = null;
  let lastFocused = null;

  if (!slides.length) {
    throw new Error("The slide document contains no section elements.");
  }

  function slideTitle(slide, index) {
    const heading = slide.querySelector("h1.title, h1:not(.subtitle), h2");
    return heading ? heading.textContent.trim() : `Slide ${index + 1}`;
  }

  slides.forEach((slide, index) => {
    slide.classList.add("deck-slide");
    if (!slide.id) slide.id = `slide-${index + 1}`;
    const title = slideTitle(slide, index);
    slide.dataset.title = title;
    slide.setAttribute("role", "group");
    slide.setAttribute("aria-roledescription", "slide");
    slide.setAttribute("aria-label", `Slide ${index + 1} of ${slides.length}: ${title}`);
    if (title.startsWith("Appendix:")) slide.classList.add("appendix");
  });

  function findIndexFromHash() {
    const raw = decodeURIComponent(window.location.hash.slice(1));
    if (!raw) return 0;
    const direct = slides.findIndex((slide) => slide.id === raw);
    if (direct >= 0) return direct;
    const numbered = raw.match(/^slide-(\d+)$/);
    if (numbered) return Math.min(slides.length - 1, Math.max(0, Number(numbered[1]) - 1));
    return 0;
  }

  function setFocusableState(slide, active) {
    slide.querySelectorAll(focusableSelector).forEach((element) => {
      if (element.closest("[role='note']")) return;
      element.tabIndex = active ? 0 : -1;
    });
  }

  function updateNotes() {
    const source = slides[currentIndex].querySelector("[role='note']");
    notesContent.innerHTML = source
      ? source.innerHTML
      : "<p>No speaker notes were provided for this slide.</p>";
  }

  function updateOverviewSelection() {
    overviewGrid.querySelectorAll(".overview-card").forEach((card, index) => {
      card.classList.toggle("current", index === currentIndex);
      card.setAttribute("aria-current", index === currentIndex ? "true" : "false");
    });
  }

  function checkOverflow(slide) {
    window.requestAnimationFrame(() => {
      const overflow = slide.scrollHeight > slide.clientHeight + 2 || slide.scrollWidth > slide.clientWidth + 2;
      slide.dataset.overflow = overflow ? "true" : "false";
      if (overflow) console.warn(`Slide overflow detected: #${slide.id}`);
    });
  }

  function showSlide(index, options = {}) {
    const nextIndex = Math.min(slides.length - 1, Math.max(0, index));
    currentIndex = nextIndex;

    slides.forEach((slide, slideIndex) => {
      const active = slideIndex === currentIndex;
      slide.classList.toggle("active", active);
      slide.setAttribute("aria-hidden", active ? "false" : "true");
      setFocusableState(slide, active);
    });

    counter.textContent = `${currentIndex + 1} / ${slides.length}`;
    progressBar.style.width = `${((currentIndex + 1) / slides.length) * 100}%`;
    document.title = `${slides[currentIndex].dataset.title} | GAVD S-JEPA`;
    updateNotes();
    updateOverviewSelection();
    liveStatus.textContent = `Slide ${currentIndex + 1} of ${slides.length}: ${slides[currentIndex].dataset.title}`;

    if (options.updateHash !== false) {
      history.replaceState(null, "", `#${slides[currentIndex].id}`);
    }
    checkOverflow(slides[currentIndex]);
  }

  function setScale() {
    const widthScale = viewport.clientWidth / 1600;
    const heightScale = viewport.clientHeight / 900;
    const scale = Math.max(0.05, Math.min(widthScale, heightScale));
    document.documentElement.style.setProperty("--deck-scale", scale.toFixed(5));
    checkOverflow(slides[currentIndex]);
  }

  function buildOverview() {
    overviewGrid.replaceChildren();
    slides.forEach((slide, index) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "overview-card";
      if (slide.classList.contains("appendix")) card.classList.add("appendix");
      card.dataset.index = String(index);
      card.innerHTML = `<span class="overview-number">${index + 1}</span><span class="overview-title"></span>`;
      card.querySelector(".overview-title").textContent = slide.dataset.title;
      card.addEventListener("click", () => {
        showSlide(index);
        toggleOverview(false);
      });
      overviewGrid.appendChild(card);
    });
  }

  function setPressed(action, pressed) {
    document.querySelectorAll(`[data-action="${action}"]`).forEach((button) => {
      if (button.hasAttribute("aria-pressed")) button.setAttribute("aria-pressed", String(pressed));
    });
  }

  function rememberFocus() {
    lastFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  }

  function restoreFocus() {
    if (lastFocused && document.contains(lastFocused)) lastFocused.focus();
    lastFocused = null;
  }

  function toggleOverview(force) {
    const open = typeof force === "boolean" ? force : overviewOverlay.hidden;
    if (open) {
      rememberFocus();
      overviewOverlay.hidden = false;
      setPressed("overview", true);
      updateOverviewSelection();
      const current = overviewGrid.querySelector(".overview-card.current");
      if (current) current.focus();
    } else {
      overviewOverlay.hidden = true;
      setPressed("overview", false);
      restoreFocus();
    }
  }

  function toggleHelp(force) {
    const open = typeof force === "boolean" ? force : helpOverlay.hidden;
    if (open) {
      rememberFocus();
      helpOverlay.hidden = false;
      setPressed("help", true);
      helpOverlay.querySelector("button").focus();
    } else {
      helpOverlay.hidden = true;
      setPressed("help", false);
      restoreFocus();
    }
  }

  function toggleNotes(force) {
    const open = typeof force === "boolean" ? force : notesPanel.hidden;
    notesPanel.hidden = !open;
    setPressed("notes", open);
    if (open) {
      updateNotes();
      notesPanel.querySelector("button").focus();
    }
  }

  async function toggleFullscreen() {
    try {
      if (!document.fullscreenElement) {
        await document.documentElement.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch (error) {
      console.warn("Fullscreen is unavailable in this browser.", error);
    }
  }

  function closeTopPanel() {
    if (!helpOverlay.hidden) return toggleHelp(false);
    if (!overviewOverlay.hidden) return toggleOverview(false);
    if (!notesPanel.hidden) return toggleNotes(false);
  }

  function runAction(action) {
    const actions = {
      previous: () => showSlide(currentIndex - 1),
      next: () => showSlide(currentIndex + 1),
      overview: () => toggleOverview(),
      notes: () => toggleNotes(),
      fullscreen: () => toggleFullscreen(),
      print: () => window.print(),
      help: () => toggleHelp(),
    };
    if (actions[action]) actions[action]();
  }

  document.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", () => runAction(button.dataset.action));
  });

  document.addEventListener("keydown", (event) => {
    const target = event.target;
    if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return;

    if (event.key === "Escape") {
      closeTopPanel();
      return;
    }
    if (!helpOverlay.hidden || !overviewOverlay.hidden) return;

    const key = event.key.toLowerCase();
    const navigation = {
      arrowright: () => showSlide(currentIndex + 1),
      pagedown: () => showSlide(currentIndex + 1),
      " ": () => showSlide(currentIndex + 1),
      arrowleft: () => showSlide(currentIndex - 1),
      pageup: () => showSlide(currentIndex - 1),
      home: () => showSlide(0),
      end: () => showSlide(slides.length - 1),
      o: () => toggleOverview(),
      n: () => toggleNotes(),
      f: () => toggleFullscreen(),
      "?": () => toggleHelp(),
    };
    if (navigation[key]) {
      event.preventDefault();
      navigation[key]();
    }
  });

  document.addEventListener("pointerdown", (event) => {
    if (event.pointerType === "mouse" || event.target.closest("a, button")) return;
    touchStart = { x: event.clientX, y: event.clientY, time: performance.now() };
  });

  document.addEventListener("pointerup", (event) => {
    if (!touchStart) return;
    const dx = event.clientX - touchStart.x;
    const dy = event.clientY - touchStart.y;
    const elapsed = performance.now() - touchStart.time;
    touchStart = null;
    if (elapsed > 900 || Math.abs(dx) < 70 || Math.abs(dx) < Math.abs(dy) * 1.25) return;
    showSlide(currentIndex + (dx < 0 ? 1 : -1));
  });

  window.addEventListener("hashchange", () => showSlide(findIndexFromHash(), { updateHash: false }));
  window.addEventListener("resize", setScale);
  document.addEventListener("fullscreenchange", setScale);

  buildOverview();
  setScale();
  showSlide(findIndexFromHash(), { updateHash: true });
})();
