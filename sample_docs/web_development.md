# Web Development Overview

Web development is the work involved in building and maintaining websites and web applications.

## Frontend Development

Frontend (client-side) development focuses on what users see and interact with in the browser.

### HTML (HyperText Markup Language)
HTML provides the structure of web pages. Key elements include:
- Headings (`<h1>` through `<h6>`)
- Paragraphs (`<p>`)
- Links (`<a href="...">`)
- Images (`<img src="...">`)
- Forms (`<form>`, `<input>`, `<button>`)
- Semantic elements (`<nav>`, `<main>`, `<article>`, `<section>`)

### CSS (Cascading Style Sheets)
CSS controls the visual presentation. Modern CSS includes:
- Flexbox for one-dimensional layouts
- CSS Grid for two-dimensional layouts
- Custom properties (CSS variables) for theming
- Media queries for responsive design
- Animations and transitions

### JavaScript
JavaScript adds interactivity to web pages. Key concepts:
- DOM manipulation for dynamic content
- Event handling for user interactions
- Fetch API for making HTTP requests
- Async/await for asynchronous operations
- ES modules for code organization

## Backend Development

Backend (server-side) development handles business logic, data storage, and API endpoints.

### Popular Backend Frameworks
- **FastAPI** (Python): Modern, fast, async-first API framework
- **Django** (Python): Full-featured web framework with ORM
- **Express** (Node.js): Minimal, flexible web framework
- **Spring Boot** (Java): Enterprise-grade application framework

### FastAPI
FastAPI is a modern Python web framework built on Starlette and Pydantic. Features include:
- Automatic OpenAPI documentation
- Type hints for request/response validation
- Async support with ASGI
- Dependency injection
- Middleware support

Example FastAPI endpoint:
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

## APIs and REST

REST (Representational State Transfer) is an architectural style for web APIs:
- **GET**: Retrieve resources
- **POST**: Create resources
- **PUT**: Update resources
- **DELETE**: Remove resources

APIs return data in JSON format and use HTTP status codes (200 OK, 404 Not Found, 500 Server Error).

## Databases

- **SQL databases**: PostgreSQL, MySQL, SQLite - structured data with relations
- **NoSQL databases**: MongoDB, Redis - flexible schemas, key-value stores
- **Vector databases**: ChromaDB, Pinecone - embeddings for similarity search
