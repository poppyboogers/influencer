// The characters this site offers, and the picker for choosing one.
// The list comes from characters/characters.json: made on the fly by the dev server, and written
// by `npm run package` for the hosted site (see scripts/characters.mjs for what a character needs).

/** Address of the folder the app is served from, whatever it is called on the host. */
export const BASE = new URL(import.meta.env.BASE_URL, document.baseURI).href;

export interface CharacterInfo {
  id: string;
  name: string;
  /** File paths inside characters/. */
  model: string;
  face?: string;
  roughness?: string;
  emissive?: string;
  thumb?: string;
  /** Changes when the character's files change. */
  v: string;
}

/** Full address of one of a character's files. */
export const characterUrl = (c: CharacterInfo, file: string) => `${BASE}characters/${file}?v=${c.v}`;

export async function loadCharacters(): Promise<CharacterInfo[]> {
  const res = await fetch(`${BASE}characters/characters.json`, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`The character list (characters/characters.json) did not load: ${res.status}`);
  return (await res.json()) as CharacterInfo[];
}

/** Address of this page showing another character, keeping the other settings (?demo and so on). */
function addressFor(id: string) {
  const url = new URL(location.href);
  url.searchParams.set('character', id);
  return url.href;
}

/**
 * Shows the character cards. With `current` set it is a panel over the running app that can be
 * closed again; without, it is the opening screen. Choosing a card reloads the page on that character.
 */
export function showPicker(characters: CharacterInfo[], current?: string): { close(): void } {
  const panel = document.createElement('div');
  panel.id = 'picker';
  const title = document.createElement('h1');
  title.textContent = 'Choose a character';
  const grid = document.createElement('div');
  grid.className = 'cards';
  for (const c of characters) {
    const card = document.createElement('a');
    card.className = 'card' + (c.id === current ? ' current' : '');
    card.href = addressFor(c.id);
    if (c.thumb) {
      const img = document.createElement('img');
      img.src = characterUrl(c, c.thumb);
      img.alt = '';
      card.append(img);
    } else {
      const blank = document.createElement('div');
      blank.className = 'blank';
      blank.textContent = c.name.charAt(0);
      card.append(blank);
    }
    const label = document.createElement('span');
    label.textContent = c.name;
    card.append(label);
    grid.append(card);
  }
  panel.append(title, grid);
  const close = () => {
    panel.remove();
    window.removeEventListener('keydown', onKey);
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && current) close();
  };
  if (current) {
    const hint = document.createElement('p');
    hint.textContent = 'Esc or click outside to go back';
    panel.append(hint);
    panel.addEventListener('click', (e) => {
      if (e.target === panel) close();
    });
    window.addEventListener('keydown', onKey);
  }
  document.body.append(panel);
  return { close };
}
