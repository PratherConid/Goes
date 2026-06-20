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
            '/api/ai': {
                target: 'http://localhost:8765',
                rewrite: path => path.replace(/^\/api\/ai/, ''),
            },
            '/api': 'http://localhost:3000',
        },
    },
});
