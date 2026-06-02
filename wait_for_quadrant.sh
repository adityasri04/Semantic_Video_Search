#!/bin/sh
echo "Waiting for Qdrant..."
until wget -qO- http://qdrant_db:6333/healthz > /dev/null; do
  sleep 2
done
echo "Qdrant is ready!"
exec "$@"   # Continue with starting the API
