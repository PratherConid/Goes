import { defineConfig } from 'vite';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
    resolve: {
        alias: {
            '@shared': fileURLToPath(new URL('./shared', import.meta.url)),
        },
    },
    server: {
        proxy: {
            // All client↔server traffic (AI proxy + online games) goes over the WS.
            '/ws': { target: 'ws://localhost:3000', ws: true },
        },
    },
});
