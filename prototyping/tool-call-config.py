labels = ["code_search", "run_tests", "open_file"]

label_descriptions = """
- code_search: search the codebase for a text or symbol query
- run_tests: run automated tests in a given path
- open_file: open a single file for inspection or editing"""

categories_types = {
    "intent": ["navigation","testing","file_access"],
}

use_case = "Map natural-language developer requests to JSON tool calls for a codebase assistant (e.g., code search, running tests, opening files)."

prompt_examples = """ 
	LABEL: code_search
	CATEGORY: intent
	TYPE: navigation
	OUTPUT: {"tool":"code_search","args":{"query":"getUserProfile","path":"backend/","max_results":50}}
	REASONING: The user wants all references to getUserProfile in backend code, so code_search with that query, backend path, and a max_results cap is appropriate.

	LABEL: run_tests
	CATEGORY: intent=testing
	TYPE: testing
	OUTPUT: {"tool":"run_tests","args":{"path":"services/auth/","test_filter":"unit"}}
	REASONING: The request is to run only unit tests for the auth service, so run_tests with the auth path and a unit test_filter is correct.

	LABEL: open_file
	CATEGORY: intent=file_access
	TYPE: file_access
	OUTPUT: {"tool":"open_file","args":{"path":"services/payments/config.yaml"}}
	REASONING: The user wants to open the main config file for the payments service, so open_file with the payments config path is the right call.
	
	LABEL: code_search
	CATEGORY: intent
	TYPE: navigation
	OUTPUT: {"tool":"code_search","args":{"query":"LegacyBillingClient","path":"src/","max_results":100}}
	REASONING: They want every reference to LegacyBillingClient for refactoring, so a broad code_search over src with that query and a higher max_results is appropriate.
	
	LABEL: run_tests
	CATEGORY: testing
	TYPE: testing
	OUTPUT: {"tool":"run_tests","args":{"path":".","test_filter":null}}
	REASONING: The instruction is to run the full test suite in CI mode, so run_tests from the repo root with no test_filter runs all tests.
            """
