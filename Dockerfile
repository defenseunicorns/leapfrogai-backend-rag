ARG ARCH=amd64

FROM ghcr.io/defenseunicorns/leapfrogai/python:3.11-dev-${ARCH} as builder

WORKDIR /leapfrogai

RUN python -m venv .venv
ENV PATH="/leapfrogai/.venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY tools/cache_tokenizer.py .
RUN python cache_tokenizer.py

FROM ghcr.io/defenseunicorns/leapfrogai/python:3.11-${ARCH}

WORKDIR /leapfrogai

ENV PATH="/leapfrogai/.venv/bin:$PATH"

COPY --from=builder /leapfrogai/tokenizer-cache/ /leapfrogai/tokenizer-cache/
COPY --from=builder /leapfrogai/.venv /leapfrogai/.venv

COPY src/ .

EXPOSE 8000

ENTRYPOINT ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]