import sys

# Skip test files whose required modules are unavailable (e.g. in mutmut sandbox)
_CONDITIONAL_TESTS = {
    "test_agent.py": "agent",
    "test_chat.py": "agent",
    "test_file_reader.py": "file_reader",
    "test_graph.py": "agent",
    "test_image_captioner.py": "image_captioner",
    "test_models.py": "models",
    "test_parser.py": "parser",
    "test_rewriter.py": "rewriter",
    "test_router.py": "router",
    "test_searcher.py": "searcher",
    "test_toolbox.py": "toolbox",
    "test_tools.py": "tools",
    "test_vision_models.py": "vision_models",
    "test_benchmark_extractors.py": "file_reader",
    "test_caption_cache.py": "image_captioner",
    "test_indexer_integration.py": "indexer",
    "test_integration.py": "agent",
}

collect_ignore = []
for test_file, module_name in _CONDITIONAL_TESTS.items():
    try:
        __import__(module_name)
    except ImportError:
        collect_ignore.append(test_file)
