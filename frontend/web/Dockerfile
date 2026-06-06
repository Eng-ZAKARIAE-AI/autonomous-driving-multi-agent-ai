# ── Stage 1: dependency cache ─────────────────────────────────────────────────
FROM node:19-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --prefer-offline

# ── Stage 2: build ────────────────────────────────────────────────────────────
FROM node:19-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# ── Stage 3: production (Nginx) ───────────────────────────────────────────────
FROM nginx:1.27-alpine AS production

# Remove default Nginx content
RUN rm -rf /usr/share/nginx/html/*

# Inject Nginx config that handles SPA routing and proxies /api + /ws to backend
COPY --chown=nginx:nginx nginx.conf /etc/nginx/conf.d/default.conf

# Copy Vite build artifacts
COPY --from=build --chown=nginx:nginx /app/dist /usr/share/nginx/html

EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:3000/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
