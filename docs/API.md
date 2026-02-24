# API Reference

**AI-SDK-CREWAI** - Complete API Documentation

---

## Base URL

```
Production: https://api.crewai.com
Development: http://localhost:8000
```

---

## Authentication

### API Key Authentication

Most endpoints require API key authentication:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.crewai.com/api/endpoint
```

---

## Endpoints

### Health Check

**GET /health**

Check service health status.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "crewai-api",
  "version": "0.1.0"
}
```

---

## Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 400 | Bad Request | Check request body format |
| 401 | Unauthorized | Verify API key is valid |
| 429 | Rate Limited | Implement backoff |
| 500 | Internal Error | Contact support |

---

For more details, see the project README.
