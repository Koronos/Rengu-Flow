import { ref } from "vue";
import { defineStore } from "pinia";

// Color themes, keyed by the `data-theme` attribute on <html> (see styles/app.css).
//   midnight — cyan accent on deep navy (default)
//   original — the classic black/neutral Element Plus dark theme
export type RenguTheme = "midnight" | "original";

const STORAGE_KEY = "rengu-flow-theme";
const THEMES: RenguTheme[] = ["midnight", "original"];
const DEFAULT_THEME: RenguTheme = "midnight";

function readStored(): RenguTheme {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v && THEMES.includes(v as RenguTheme)) return v as RenguTheme;
  } catch {
    /* localStorage may be unavailable (private mode); fall back to default */
  }
  return DEFAULT_THEME;
}

function applyTheme(t: RenguTheme): void {
  document.documentElement.dataset.theme = t;
}

export const useThemeStore = defineStore("theme", () => {
  const theme = ref<RenguTheme>(readStored());

  function setTheme(t: RenguTheme): void {
    theme.value = t;
    try {
      localStorage.setItem(STORAGE_KEY, t);
    } catch {
      /* ignore persistence failures */
    }
    applyTheme(t);
  }

  // Keep the live DOM in sync with the stored value when the store first initializes (an inline
  // script in index.html applies it pre-paint to avoid a flash; this re-applies for SPA reactivity).
  applyTheme(theme.value);

  return { theme, setTheme };
});
