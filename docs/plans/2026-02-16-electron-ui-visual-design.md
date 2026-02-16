# NeuroFind Visual Design System

**Date:** 2026-02-16
**Status:** Proposed
**Aesthetic:** Warm Brutalism — a precision instrument, not a SaaS product

## Concept

NeuroFind is a local, offline AI file assistant. It runs on your hardware, processes your documents, never phones home. The UI should reflect this: **intimate, trustworthy, mechanical**. Like a well-made desk tool you use every day.

The memorable signature: watching the agent *think*. Tool calls animate in with a satisfying mechanical reveal — gears turning, not magic happening.

## Aesthetic Direction

**Warm Brutalism** — dark, warm tones with exposed structure. The interface doesn't hide what it's doing. You see the agent's reasoning steps. File paths and tool names are displayed proudly in monospace. But the conversation itself is warm, readable, human.

**NOT:** Cold blue-gray tech aesthetic. NOT: Purple gradient SaaS. NOT: Bubbly chat app.

**YES:** Dark walnut tones. Amber accents. Exposed monospace traces. Confident typography. Subtle grain texture. The feeling of a leather notebook open next to a terminal.

## Typography

Three fonts, each with a clear role:

| Role | Font | Weight | Usage |
|------|------|--------|-------|
| Display | **Cormorant Garamond** | 600-700 | App title, headings, welcome screen |
| Body | **DM Sans** | 400-500 | Chat messages, UI labels, buttons |
| Mono | **IBM Plex Mono** | 400 | File paths, tool names, agent traces, code |

Import via Google Fonts:
```css
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=DM+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
```

## Color Palette

Dark theme with warm undertones. No pure black — everything has a brown/amber warmth.

```css
:root {
  /* Base */
  --bg-primary: #1a1714;          /* warm near-black */
  --bg-secondary: #231f1b;        /* slightly lifted */
  --bg-tertiary: #2d2723;         /* cards, panels */
  --bg-elevated: #38312b;         /* hover states, active */

  /* Text */
  --text-primary: #e8e0d6;        /* warm off-white */
  --text-secondary: #a89b8c;      /* muted, labels */
  --text-tertiary: #6d6259;       /* hints, timestamps */

  /* Accent */
  --accent: #d4915a;              /* warm amber — primary action */
  --accent-hover: #e0a36e;        /* lighter amber */
  --accent-muted: #d4915a33;      /* amber at 20% for backgrounds */

  /* Semantic */
  --success: #7ab87a;             /* muted green */
  --warning: #d4a24e;             /* gold */
  --error: #c45c5c;               /* muted red */

  /* Borders */
  --border: #38312b;              /* subtle, warm */
  --border-active: #4d443c;       /* hover/focus */

  /* Effects */
  --grain-opacity: 0.03;          /* subtle film grain texture */
  --shadow-color: rgba(10, 8, 6, 0.5);
}
```

## Background Texture

A subtle film grain overlay on the root element. Creates depth without being distracting.

```css
.app-root::before {
  content: '';
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 9999;
  opacity: var(--grain-opacity);
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
}
```

## Layout

```
┌──────────────────────────────────────────────────┐
│  NeuroFind          [folder path]     [debug] [⚙] │  ← Header (48px)
├──────────────────────────────────────────────────┤
│                                                  │
│                                                  │
│    Message bubbles with streaming text            │
│    Agent step traces (collapsible)               │  ← Chat area (flex-1)
│                                                  │
│                                                  │
│                                                  │
├──────────────────────────────────────────────────┤
│  [icon] Ask about your files...          [Send]  │  ← Input (64px)
├──────────────────────────────────────────────────┤
│  ● Ready  │  LFM2.5-1.2B  │  ~/Documents       │  ← Status (32px)
└──────────────────────────────────────────────────┘
```

Single-column chat-first layout. No sidebar by default. The header shows the current folder path as a clickable breadcrumb to change directories.

## Component Designs

### Header

Minimal. App title in Cormorant Garamond (serif) on the left — establishes the warm, personal tone immediately. Current folder path in IBM Plex Mono on the right, clickable to change directory.

```tsx
// Tailwind classes (conceptual)
<header className="flex items-center justify-between h-12 px-5 border-b border-[--border] bg-[--bg-secondary]">
  <h1 className="font-display text-xl font-semibold text-[--text-primary] tracking-tight">
    NeuroFind
  </h1>
  <button className="font-mono text-xs text-[--text-tertiary] hover:text-[--accent] transition-colors">
    ~/Documents/invoices
  </button>
</header>
```

### Chat Messages

**User messages:** Right-aligned, amber accent background (muted), rounded corners with a flat bottom-right to indicate direction. DM Sans.

**Assistant messages:** Left-aligned, bg-tertiary background, rounded corners with flat bottom-left. Streaming text appears character by character with a blinking amber cursor at the end.

```tsx
// User message
<div className="ml-auto max-w-[75%] rounded-2xl rounded-br-md bg-[--accent-muted] px-4 py-3 border border-[--accent]/20">
  <p className="text-sm text-[--text-primary]">{message.text}</p>
</div>

// Assistant message
<div className="mr-auto max-w-[75%] rounded-2xl rounded-bl-md bg-[--bg-tertiary] px-4 py-3 border border-[--border]">
  <p className="text-sm text-[--text-primary] whitespace-pre-wrap">{message.text}</p>
  {message.isStreaming && (
    <span className="inline-block w-0.5 h-4 bg-[--accent] animate-pulse ml-0.5" />
  )}
</div>
```

### Agent Steps (The Signature Element)

When the agent calls tools, they appear as a compact trace *inside* the assistant message bubble. Each step slides in from the left with a staggered delay, like a typewriter printing. The tool name is in a monospace badge, parameters beside it.

**Animation:** Each step enters with `translateX(-8px) → 0` and `opacity: 0 → 1` over 200ms, staggered by 80ms. A thin amber line connects the steps vertically on the left edge.

```tsx
// Agent step trace
<div className="mt-3 border-t border-[--border] pt-2">
  <button className="text-[10px] font-mono text-[--text-tertiary] uppercase tracking-widest hover:text-[--text-secondary]">
    {steps.length} steps
  </button>

  {/* Expanded: */}
  <div className="relative mt-2 ml-2 pl-3 border-l border-[--accent]/30">
    {steps.map((step, i) => (
      <motion.div
        key={i}
        initial={{ opacity: 0, x: -8 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: i * 0.08, duration: 0.2 }}
        className="flex items-center gap-2 py-1"
      >
        {/* Dot on the timeline */}
        <span className="absolute -left-[5px] h-2 w-2 rounded-full bg-[--accent]" />

        <span className="font-mono text-[11px] text-[--accent] bg-[--accent]/10 px-1.5 py-0.5 rounded">
          {step.tool}
        </span>
        <span className="font-mono text-[11px] text-[--text-tertiary] truncate">
          {formatParams(step.params)}
        </span>
      </motion.div>
    ))}
  </div>
</div>
```

### Streaming Text Animation

Tokens arrive from the backend. Rather than snapping in, each token fades in with a micro-animation. The overall effect is smooth flowing text, not jarring chunks.

We use a CSS approach — new tokens are wrapped in a `<span>` with a fade-in animation:

```css
@keyframes token-in {
  from { opacity: 0.4; }
  to { opacity: 1; }
}

.token-new {
  animation: token-in 150ms ease-out;
}
```

The cursor (amber blinking bar) always sits at the end of the streaming text. When streaming completes, it fades out over 300ms.

### Input Bar

A single-line input with a warm, inset feel — like typing into a field carved from wood. Subtle inner shadow. The send button is a filled amber circle with an arrow icon, disabled (grayed) when input is empty or backend is loading.

```tsx
<form className="flex items-center gap-3 px-5 py-3 border-t border-[--border] bg-[--bg-secondary]">
  <div className="flex-1 relative">
    <input
      className="w-full bg-[--bg-primary] text-sm text-[--text-primary] placeholder:text-[--text-tertiary]
                 rounded-xl px-4 py-3 border border-[--border] focus:border-[--accent]/50 focus:outline-none
                 shadow-[inset_0_1px_3px_rgba(0,0,0,0.3)] transition-colors"
      placeholder="Ask about your files..."
    />
  </div>
  <button
    className="h-10 w-10 rounded-full bg-[--accent] text-[--bg-primary] flex items-center justify-center
               hover:bg-[--accent-hover] disabled:opacity-30 disabled:cursor-not-allowed
               transition-all active:scale-95"
  >
    <ArrowUpIcon className="h-4 w-4" />
  </button>
</form>
```

### Status Bar

Minimal, monospace, at the very bottom. Three segments separated by thin vertical dividers: connection status (colored dot + label), model name, current directory.

```tsx
<div className="flex items-center gap-0 h-8 px-5 border-t border-[--border] bg-[--bg-primary] font-mono text-[11px] text-[--text-tertiary]">
  <div className="flex items-center gap-1.5 pr-4 border-r border-[--border]">
    <span className={`h-1.5 w-1.5 rounded-full ${statusColor}`} />
    <span>{statusLabel}</span>
  </div>
  <div className="px-4 border-r border-[--border]">LFM2.5-1.2B</div>
  <div className="px-4 truncate">{dataDir}</div>
</div>
```

### Welcome Screen (FileBrowser / First Run)

When no directory is selected, the chat area shows a centered welcome. Cormorant Garamond for the headline, DM Sans for the description. A single large "Open Folder" button in amber.

The headline animates in on mount: fade up from 20px below over 600ms with a slight blur clearing.

```tsx
<motion.div
  initial={{ opacity: 0, y: 20, filter: "blur(4px)" }}
  animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
  transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
  className="flex flex-col items-center justify-center h-full gap-6"
>
  <h2 className="font-display text-4xl font-bold text-[--text-primary] tracking-tight">
    Your files, your AI.
  </h2>
  <p className="text-[--text-secondary] text-center max-w-md leading-relaxed">
    NeuroFind runs entirely on your machine. Select a folder to index
    and start asking questions about your documents.
  </p>
  <Button size="lg" className="bg-[--accent] hover:bg-[--accent-hover] text-[--bg-primary] font-medium px-8 py-3 rounded-xl">
    Open Folder
  </Button>
</motion.div>
```

### Loading States

**Model loading:** Full-screen centered. The NeuroFind title pulses gently. Below it, a status line in monospace cycles through states: "Loading model..." → "Warming up..." → "Indexing files..." with a smooth crossfade.

**Query loading (after send, before first token):** Three amber dots pulse in sequence inside the assistant message bubble. Disappears as soon as the first token arrives.

```tsx
// Typing indicator
<div className="flex gap-1 py-2 px-1">
  {[0, 1, 2].map((i) => (
    <motion.span
      key={i}
      className="h-1.5 w-1.5 rounded-full bg-[--accent]"
      animate={{ opacity: [0.3, 1, 0.3] }}
      transition={{ duration: 1, repeat: Infinity, delay: i * 0.15 }}
    />
  ))}
</div>
```

### Transitions

| Event | Animation | Duration | Easing |
|-------|-----------|----------|--------|
| Message enters (user) | Slide up + fade in from right | 300ms | ease-out |
| Message enters (assistant) | Slide up + fade in from left | 300ms | ease-out |
| Token streams | Opacity 0.4 → 1 per token | 150ms | ease-out |
| Streaming cursor | Pulse (opacity 1 → 0.3) | 800ms | ease-in-out, infinite |
| Streaming ends | Cursor fade out | 300ms | ease-out |
| Agent step enters | Slide in from left + fade | 200ms | ease-out, stagger 80ms |
| Agent steps expand | Height auto + fade | 250ms | ease-out |
| Agent steps collapse | Height 0 + fade | 200ms | ease-in |
| Welcome screen mount | Fade up + blur clear | 600ms | [0.22, 1, 0.36, 1] |
| Status change (dot color) | Color transition | 500ms | ease |
| Error banner | Slide down from top | 250ms | ease-out |
| Error dismiss | Slide up + fade | 200ms | ease-in |
| Button press | Scale 0.95 | 100ms | ease |
| Input focus | Border color transition | 200ms | ease |

### Scroll Behavior

Chat auto-scrolls to bottom on new messages/tokens, but only if the user is already near the bottomb (within 100px). If user scrolled up to read history, don't auto-scroll — show a "New message" pill at the bottom that scrolls down on click.

```tsx
// Scroll-to-bottom pill
<motion.button
  initial={{ opacity: 0, y: 10 }}
  animate={{ opacity: 1, y: 0 }}
  exit={{ opacity: 0, y: 10 }}
  className="absolute bottom-20 left-1/2 -translate-x-1/2 bg-[--accent] text-[--bg-primary]
             text-xs font-medium px-3 py-1.5 rounded-full shadow-lg"
  onClick={scrollToBottom}
>
  New message
</motion.button>
```

## shadcn Theme Overrides

Override shadcn's CSS variables to match our palette:

```css
@layer base {
  :root {
    --background: 18 15% 9%;          /* --bg-primary */
    --foreground: 30 20% 88%;          /* --text-primary */
    --card: 20 15% 12%;               /* --bg-tertiary */
    --card-foreground: 30 20% 88%;
    --popover: 20 15% 12%;
    --popover-foreground: 30 20% 88%;
    --primary: 27 60% 59%;            /* --accent */
    --primary-foreground: 18 15% 9%;
    --secondary: 20 12% 18%;          /* --bg-elevated */
    --secondary-foreground: 30 20% 88%;
    --muted: 20 12% 18%;
    --muted-foreground: 25 12% 55%;   /* --text-secondary */
    --accent: 27 60% 59%;
    --accent-foreground: 18 15% 9%;
    --destructive: 0 50% 50%;
    --destructive-foreground: 30 20% 88%;
    --border: 20 12% 18%;
    --input: 20 12% 18%;
    --ring: 27 60% 59%;
    --radius: 0.75rem;
  }
}
```

## Tailwind Config Extensions

```js
// tailwind.config.js additions
{
  theme: {
    extend: {
      fontFamily: {
        display: ['"Cormorant Garamond"', 'Georgia', 'serif'],
        sans: ['"DM Sans"', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'Menlo', 'monospace'],
      },
      keyframes: {
        'token-in': {
          from: { opacity: '0.4' },
          to: { opacity: '1' },
        },
      },
      animation: {
        'token-in': 'token-in 150ms ease-out',
      },
    },
  },
}
```

## Motion (framer-motion) Usage

Install: `npm i motion`

Import: `import { motion, AnimatePresence } from "motion/react"`

Used for:
- Message entry animations (layout animations)
- Agent step staggered reveals
- Welcome screen entrance
- Typing indicator dots
- Scroll-to-bottom pill
- Error banner slide

**Not** used for: token streaming (CSS-only for performance), cursor pulse (CSS), hover states (CSS transitions).

## Accessibility

- All interactive elements have visible focus rings (amber ring, 2px offset)
- Chat messages use `role="log"` and `aria-live="polite"` for screen reader announcements
- Agent steps toggle uses `aria-expanded`
- Color contrast: all text passes WCAG AA on dark backgrounds
- Input has proper `aria-label` when placeholder is the only label
- Status indicators have text labels alongside color dots (not color-only)

## Summary

The visual design prioritizes:
1. **Warmth** — dark amber palette, serif display font, film grain texture
2. **Transparency** — visible agent reasoning, monospace traces, exposed tool calls
3. **Polish** — smooth streaming, staggered animations, micro-interactions
4. **Restraint** — single-column layout, no unnecessary chrome, generous spacing
