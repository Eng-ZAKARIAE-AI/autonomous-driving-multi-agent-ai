FROM node:18-alpine

WORKDIR /app

# In a real scenario, we would copy package.json and install deps
# For now, we'll just provide a placeholder to satisfy docker-compose
RUN echo "Frontend placeholder" > index.html

EXPOSE 3000

CMD ["npx", "serve", "-s", ".", "-p", "3000"]
