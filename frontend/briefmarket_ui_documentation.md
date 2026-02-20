# BriefMarket AI — Frontend UI Documentation

> **Updated:** Feb 18, 2026 | **Scope:** Frontend only | **Stack:** Vanilla HTML + CSS + JS (~93 lines)

---

## What This UI Does

A static 4-page editorial prototype delivering daily AI-generated stock briefs. Users scan signal labels, confidence scores, and narrative momentum — then expand details on stocks they follow. All data is hardcoded in HTML; no API calls.

**Pages:** Today (main feed) · Collections · Watchlist · About

---

## Controls by Screen

### Global (All Pages)

| Control | Type | Wired? | Behavior |
|---------|------|--------|----------|
| **Nav Links** (Today / Collections / Watchlist / About) | Link ×4 | ✅ | Full page nav. `aria-current="page"` on active. |
| **Ticker Search** | `<input type="search">` | ⚠️ No | Appears on all pages. No JS handler — purely visual. |
| **Popular Ticker Chips** (NVDA, AAPL, MSFT, TSLA, AMZN) | Static span ×5 | ⚠️ No | Not clickable. `aria-hidden="true"`. Decorative. |
| **Footer: Disclaimer / Contact** | Link | ⚠️ No | `href="#"` — no destination. |

---

### Today Page — Feed Controls

| Control | Type | Wired? | Behavior |
|---------|------|--------|----------|
| **Brief Search** | `<input type="search">` | ⚠️ No | Placeholder says "visual only". No JS. |
| **Category Chips** (All, Bullish, Bearish, High Volatility, Earnings, Macro, Rumors) | Button ×7 | ⚠️ No | No `data-filter-*` attrs, no JS listener. Visual only. |
| **Signal Strength Filter** (All, Bullish 70%+, Mixed, Bearish, Low Confidence) | Button ×5 | ✅ | Single-select. Hides non-matching articles via `data-signal`. AND-combined with Narrative filter. |
| **Narrative Momentum Filter** (All, Strengthening, Weakening, Reversal, Fresh Coverage) | Button ×5 | ✅ | Single-select. Hides non-matching articles via `data-narrative`. AND-combined with Signal filter. |
| **View Toggle** (Cards / Compact) | Radio ×2 | ✅ (CSS) | Compact hides summaries, expand labels, and details. No JS needed. |
| **Beginner Mode** | Checkbox toggle | ✅ | Shows explainer panel + swaps all jargon `.eli15` spans to plain-English (`data-simple`). Reversible. |

---

### Today Page — Per-Article Controls (×17 articles)

| Control | Type | Wired? | Behavior |
|---------|------|--------|----------|
| **AI Confidence pill** | Clickable badge | ✅ | Toggles confidence breakdown panel (positive/negative drivers). Animated 320ms. `aria-expanded` updated. |
| **Show Key Points** | Label + hidden checkbox | ✅ (CSS) | Reveals bullet takeaways, "what changed", source badges. Hidden in Compact view. |
| **Stock Context** | Button | ✅ | Expands panel: trend, valuation, earnings date, volatility, sector momentum. |
| **Narrative Timeline** | Button | ✅ | Expands purple-tinted timeline: monthly narrative evolution (Jan→Apr). |

> All four per-article panels can open simultaneously — no mutual exclusion.

---

### Collections / Watchlist / About

No interactive controls beyond global nav and search. Content is static cards:
- **Collections:** 6 theme packs (Starter Pack, AI Core, etc.) — no click-through
- **Watchlist:** 8 ticker cards with confidence % and price change — display only, no add/remove
- **About:** 4 info cards — purely informational

---

## Filters Spec

| Filter | Values | Default | Select | Reset | Combined |
|--------|--------|---------|--------|-------|----------|
| **Signal Strength** | All, Bullish(70%+), Mixed, Bearish, Low Confidence | All | Single | Click "All" | AND with Narrative |
| **Narrative Momentum** | All, Strengthening, Weakening, Reversal, Fresh Coverage | All | Single | Click "All" | AND with Signal |

- **No "Clear All" button.** Must reset each filter individually.
- **No applied-filter badges** shown outside the chip rows.
- **No sort controls** exist. Article order is hardcoded by publish time.

---

## Critical Gaps

| Issue | Impact |
|-------|--------|
| **No empty state** — when all articles filtered out, blank area with no message | Users think the page is broken |
| **Search bars not wired** (both Ticker + Brief) on any page | #1 discovery tool does nothing |
| **Category chips not wired** (top row) | Most visible filter does nothing |
| **No state persistence** — filters/toggles reset on page reload | Frustrating on navigation |
| **Footer Disclaimer/Contact link to `#`** | Dead links |

---

## Accessibility Summary

**Good:** `aria-label` on all sections, `aria-expanded` on toggles, `aria-current="page"` on nav, `role="group"` on chip rows, `focus-visible` outlines, semantic HTML throughout.

**Gaps:** View toggle missing `role="tabpanel"` on target · "Show key points" label doesn't change text on expand · No skip-to-content link · Category chips lack `aria-pressed`.

---

## Top 3 Priorities

| # | What | Why | Effort |
|---|------|-----|--------|
| **1** | **Wire Search Bar** — filter articles by ticker/title/sector as user types | Search exists on every page doing nothing. Beginners try search first. Trust-breaking when it's dead. | **S** — ~30 lines JS, extend existing filter logic |
| **2** | **Wire Category Chips** — make top-row filter functional | Most prominent filter position. Users click these before Signal/Narrative filters. Silent failure. | **S** — ~20 lines JS + add `data-filter-category` attrs |
| **3** | **Add Empty-State Message** — show "No briefs match your filters" + clear button | Blank screen when all filtered = confusion. Beginners need reassurance the system works. | **S** — ~15 lines JS + 1 hidden div |
