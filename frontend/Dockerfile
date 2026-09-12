# Multi-stage Dockerfile for Hiver AI Recruiter-Facing Frontend
# Stage 1: Build production React bundle
FROM node:18-alpine AS builder

WORKDIR /app

# Copy dependency manifests
COPY package.json package-lock.json ./
RUN npm ci

# Copy source code and build production SPA
COPY . .

# Set API base URL for Vite build (browser connects to local backend on port 8000)
ENV VITE_API_BASE_URL=http://localhost:8000/api/v1

RUN npm run build

# Stage 2: Serve static production assets with Nginx
FROM nginx:alpine AS runner

# Copy custom Nginx configuration listening on port 3000
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built SPA static assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
