# `get_langchain_chat_open_ai_client()` does not configure async HTTP client, causing `ainvoke()` to fail with 401

## Summary

`WorkspaceClient().serving_endpoints.get_langchain_chat_open_ai_client()` only configures the sync HTTP client with Databricks authentication. Async operations (`ainvoke`, `astream`, etc.) use an unauthenticated default client and fail with HTTP 401.

This blocks use of LangGraph agents and any LangChain pattern that uses async LLM calls.

## Reproduction

```python
import asyncio
from langchain_core.messages import HumanMessage
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
llm = w.serving_endpoints.get_langchain_chat_open_ai_client(model='my-endpoint')

# Sync: works
llm.invoke([HumanMessage(content='ping')])

# Async: fails with 401
asyncio.run(llm.ainvoke([HumanMessage(content='ping')]))
```

**Expected**: Both calls succeed with identical authentication.

**Actual**: `invoke()` returns 200 OK, `ainvoke()` returns 401 Unauthorized.

## Root Cause

`get_langchain_chat_open_ai_client()` passes `http_client` but not `http_async_client`:

```python
# databricks/sdk/mixins/open_ai_client.py:105-110
return ChatOpenAI(
    model=model,
    openai_api_base=self._api._cfg.host + "/serving-endpoints",
    api_key="no-token",
    http_client=self._get_authorized_http_client(),
    # http_async_client is NOT set
)
```

LangChain's `ChatOpenAI` maintains separate HTTP clients:
- `http_client`: Used by `invoke()`, `stream()` (sync methods)
- `http_async_client`: Used by `ainvoke()`, `astream()` (async methods)

When `http_async_client` is unset, LangChain creates a default `httpx.AsyncClient` without Databricks authentication headers.

## Impact

This affects any code that uses async LangChain patterns with Databricks model serving:

- **LangGraph agents**: `create_react_agent()` and similar use `ainvoke()` internally
- **Async streaming**: `astream()`, `astream_events()`
- **Batch async**: `abatch()`
- **Any custom async chains**

## Proposed Fix

Add `http_async_client` parameter using an async-compatible authenticated client:

```python
def _get_authorized_async_http_client(self):
    import httpx

    # BearerAuth works for both sync and async httpx clients
    databricks_token_auth = BearerAuth(self._api._cfg.authenticate)
    return httpx.AsyncClient(auth=databricks_token_auth)

def get_langchain_chat_open_ai_client(self, model: str, **kwargs):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        openai_api_base=self._api._cfg.host + "/serving-endpoints",
        api_key="no-token",
        http_client=self._get_authorized_http_client(),
        http_async_client=self._get_authorized_async_http_client(),  # ADD THIS
        **kwargs,
    )
```

The existing `BearerAuth` class (lines 17-26) already works with `httpx.AsyncClient` since it uses the generator-based `auth_flow()` pattern.

## Workaround

Until this is fixed, manually construct `ChatOpenAI` with both clients:

```python
import httpx
from databricks.sdk import WorkspaceClient
from langchain_openai import ChatOpenAI

class BearerAuth(httpx.Auth):
    def __init__(self, authenticate_fn):
        self._authenticate = authenticate_fn

    def auth_flow(self, request: httpx.Request):
        headers = self._authenticate()
        request.headers["Authorization"] = headers["Authorization"]
        yield request

w = WorkspaceClient()
auth = BearerAuth(w.config.authenticate)

llm = ChatOpenAI(
    model="my-endpoint",
    openai_api_base=f"{w.config.host}/serving-endpoints",
    api_key="no-token",
    http_client=httpx.Client(auth=auth),
    http_async_client=httpx.AsyncClient(auth=auth),
)
```

## Environment

- databricks-sdk: 0.76.0
- langchain-openai: 1.1.6
- Python: 3.12

## Related

- #847 - Feature request for `AsyncOpenAI` client helper (same underlying issue for the OpenAI client)
