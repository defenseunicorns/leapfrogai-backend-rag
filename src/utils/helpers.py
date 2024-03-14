import logging
import threading
from functools import partial
from requests import HTTPError
from fastapi import FastAPI
from fastapi.middleware import Middleware

from document_store import DocumentStore


class RaisingThread(threading.Thread):
  """Overrides the original behavior of Threads to propagate exceptions to the main thread"""
  def run(self):
    self._exc = None
    try:
      super().run()
    except Exception as e:
      self._exc = e

  def join(self, timeout=None):
    super().join(timeout=timeout)
    if self._exc:
      raise self._exc
    
def remove_middleware(app: FastAPI, target: str) -> FastAPI:
    new_middlewares: list[Middleware] = []
    for middleware in app.user_middleware:
        if not middleware.cls.__name__ == target:
            new_middlewares.append(middleware)
    app.user_middleware = new_middlewares
    app.middleware_stack = app.build_middleware_stack()
    return app