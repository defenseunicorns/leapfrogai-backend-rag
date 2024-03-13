import logging
import threading
from functools import partial
from requests import HTTPError

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

def upstream_health_check(doc_store: DocumentStore):
    """Check the upstream OpenAI compatible API prior to evaluating certain requests"""
    def _pseudo_decorator(func):
        def wrapper(*args, **kwargs):
            """Perform the actual health check"""
            response = doc_store.api_healthcheck()
            if response.status_code != 200:
                logging.error("Upstream health check has failed: {}".format(response.json()))
                raise HTTPError(response=response)
            else:
                logging.debug("Upstream health check has finished successfully")
                func(*args, **kwargs)
        
        return wrapper
    return _pseudo_decorator