import { execFile } from 'node:child_process';
import fs from 'node:fs';
import { defineConfig, type Plugin } from 'vite';
import { listCharacters } from './scripts/characters.mjs';

/**
 * Lets the model viewer save joint positions: POST /api/joints/<id> with { name: [x, y, z] | null }
 * merges them into characters/<id>/<id>.joints.json (null removes an edit) and rebuilds the rigged model.
 */
function jointEditor(): Plugin {
  return {
    name: 'joint-editor',
    configureServer(server) {
      server.middlewares.use('/api/joints', (req, res) => {
        const id = (req.url ?? '').replace(/^\//, '').split('?')[0];
        const reply = (status: number, body: unknown) => {
          res.statusCode = status;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify(body));
        };
        if (req.method !== 'POST' || !/^[\w-]+(\.[\w-]+)*$/.test(id) || !fs.existsSync(`characters/${id}/${id}.rig.json`)) return reply(400, { ok: false, log: 'Unknown character' });
        let text = '';
        req.on('data', (chunk) => (text += chunk));
        req.on('end', () => {
          try {
            const file = `characters/${id}/${id}.joints.json`;
            const joints = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : {};
            for (const [name, pos] of Object.entries(JSON.parse(text) as Record<string, number[] | null>)) {
              if (pos === null) delete joints[name];
              else if (Array.isArray(pos) && pos.length === 3 && pos.every(Number.isFinite)) joints[name] = pos.map((v) => +v.toFixed(4));
            }
            fs.writeFileSync(file, JSON.stringify(joints, null, 2));
            execFile(process.execPath, ['scripts/rig-character.mjs', id], (error, stdout, stderr) => {
              reply(error ? 500 : 200, { ok: !error, log: stdout + stderr });
            });
          } catch (err) {
            reply(400, { ok: false, log: String(err) });
          }
        });
      });
    },
  };
}

/**
 * Dev server only: answers characters/characters.json with the characters found in the folder
 * (the hosted site gets a real file from `npm run package`), and reloads the app when a
 * character's face tuning file is saved.
 */
function characterList(): Plugin {
  return {
    name: 'character-list',
    configureServer(server) {
      server.middlewares.use('/characters/characters.json', (_req, res) => {
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Cache-Control', 'no-cache');
        res.end(JSON.stringify(listCharacters('characters')));
      });
      server.watcher.on('change', (file) => {
        if (file.endsWith('.face.json')) server.ws.send({ type: 'full-reload' });
      });
    },
  };
}

export default defineConfig({
  plugins: [jointEditor(), characterList()],
  server: {
    watch: {
      // The Python environment and model cache live in tools/. Watching them is slow and,
      // on Windows, locks directories while they are being installed.
      ignored: ['**/tools/**', '**/dist/**', '**/site/**'],
    },
  },
});
