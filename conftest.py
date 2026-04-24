import sys
import os

sys.path.insert(0, os.path.abspath("."))

# ChatOpenAI / OpenAIEmbeddings check for an API key at construction time.
# Tests must not hit the network, but modules are imported at collection.
# Set a dummy key so construction succeeds; test mocks guard the actual calls.
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")
