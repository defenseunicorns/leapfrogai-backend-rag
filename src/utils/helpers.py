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