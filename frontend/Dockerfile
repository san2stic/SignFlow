FROM node:20-alpine

WORKDIR /app

COPY package.json /app/
COPY tsconfig.json /app/
COPY vite.config.ts /app/
COPY tailwind.config.ts /app/
COPY postcss.config.js /app/
COPY index.html /app/
COPY public /app/public
COPY src /app/src

RUN npm install

EXPOSE 3000
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]
