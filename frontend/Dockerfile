# Build-Stage: Angular mit gepinntem Node-LTS bauen (lokales Node v26 wird NICHT genutzt).
FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

# Serve-Stage: statische Dateien via nginx + /api-Reverse-Proxy (SSE-tauglich).
FROM nginx:1.27-alpine
ENV BACKEND_UPSTREAM=backend:8000
COPY default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist/frontend/browser /usr/share/nginx/html
EXPOSE 80
