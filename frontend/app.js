/* =========================================================
   BriefMarket AI – Interactive Enhancements (vanilla JS)
   ========================================================= */

document.addEventListener('DOMContentLoaded', () => {

  /* -------------------------------------------------------
     1. Confidence Breakdown Toggle
     ------------------------------------------------------- */
  document.querySelectorAll('.confidence[data-toggle]').forEach(pill => {
    pill.addEventListener('click', () => {
      const panel = pill.closest('.article__signals')
        .nextElementSibling;
      if (panel && panel.classList.contains('confidence-breakdown')) {
        panel.classList.toggle('is-open');
        pill.setAttribute('aria-expanded',
          panel.classList.contains('is-open'));
      }
    });
  });

  /* -------------------------------------------------------
     2. Generic collapsible toggle (Stock Context + Narrative)
     ------------------------------------------------------- */
  document.querySelectorAll('[data-collapse]').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-collapse');
      const panel = document.getElementById(targetId);
      if (!panel) return;
      panel.classList.toggle('is-open');
      btn.setAttribute('aria-expanded',
        panel.classList.contains('is-open'));
    });
  });

  /* -------------------------------------------------------
     3. Signal Strength Filter
     ------------------------------------------------------- */
  const signalChips = document.querySelectorAll('[data-filter-signal]');
  signalChips.forEach(chip => {
    chip.addEventListener('click', () => {
      signalChips.forEach(c => c.classList.remove('chip--active'));
      chip.classList.add('chip--active');
      applyFilters();
    });
  });

  /* -------------------------------------------------------
     4. Narrative Momentum Filter
     ------------------------------------------------------- */
  const narrativeChips = document.querySelectorAll('[data-filter-narrative]');
  narrativeChips.forEach(chip => {
    chip.addEventListener('click', () => {
      narrativeChips.forEach(c => c.classList.remove('chip--active'));
      chip.classList.add('chip--active');
      applyFilters();
    });
  });

  function applyFilters() {
    const activeSignal = document.querySelector('[data-filter-signal].chip--active');
    const activeNarrative = document.querySelector('[data-filter-narrative].chip--active');
    const sigVal = activeSignal ? activeSignal.getAttribute('data-filter-signal') : 'all';
    const narVal = activeNarrative ? activeNarrative.getAttribute('data-filter-narrative') : 'all';

    document.querySelectorAll('.article[data-signal]').forEach(card => {
      const matchSig = sigVal === 'all' || card.getAttribute('data-signal') === sigVal;
      const matchNar = narVal === 'all' || card.getAttribute('data-narrative') === narVal;
      card.classList.toggle('article--hidden', !(matchSig && matchNar));
    });
  }

  /* -------------------------------------------------------
     5. Beginner Mode – also swaps jargon to plain English
     ------------------------------------------------------- */
  const beginner = document.getElementById('beginner-mode');
  if (beginner) {
    beginner.addEventListener('change', () => {
      document.querySelectorAll('.eli15').forEach(span => {
        if (beginner.checked) {
          span.dataset.original = span.textContent;
          span.textContent = span.dataset.simple;
          span.classList.add('eli15--active');
        } else {
          span.textContent = span.dataset.original;
          span.classList.remove('eli15--active');
        }
      });
    });
  }

});
