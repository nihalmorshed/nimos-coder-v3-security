"""
Security Test Suite for Nimo's Coder Agent v3

Run this after training to evaluate security capabilities.
"""

import sys
sys.path.append('..')

from src.inference import CoderAgentV3


# Test cases
SECURITY_TESTS = [
    {
        "name": "Command Injection Detection",
        "code": '''import os
user_input = input("Enter filename: ")
os.system(f"cat {user_input}")''',
        "instruction": "Review this code for security vulnerabilities. Is it safe?",
        "expected_keywords": ["vulnerability", "injection", "unsafe", "dangerous", "no", "os.system"],
        "should_fail": True,  # Code is vulnerable
    },
    {
        "name": "SQL Injection Detection",
        "code": 'query = f"SELECT * FROM users WHERE id = {user_id}"',
        "instruction": "Is this code secure?",
        "expected_keywords": ["sql injection", "parameterized", "unsafe", "no", "vulnerable"],
        "should_fail": True,
    },
    {
        "name": "Safe Parameterized Query",
        "code": 'cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))',
        "instruction": "Is this code secure?",
        "expected_keywords": ["safe", "secure", "yes", "correct", "parameterized"],
        "should_fail": False,
    },
    {
        "name": "XSS Detection",
        "code": 'document.innerHTML = userInput;',
        "instruction": "Review this JavaScript code for security issues",
        "expected_keywords": ["xss", "cross-site", "sanitize", "unsafe", "escape"],
        "should_fail": True,
    },
    {
        "name": "Path Traversal Detection",
        "code": '''filename = request.args.get('file')
with open(f'/data/{filename}', 'r') as f:
    return f.read()''',
        "instruction": "Is this code safe?",
        "expected_keywords": ["path traversal", "directory", "unsafe", "../", "validate"],
        "should_fail": True,
    },
]

ERROR_HANDLING_TESTS = [
    {
        "name": "File Reading Function",
        "instruction": "Write a Python function to read a file",
        "expected_keywords": ["try", "except", "FileNotFoundError", "open"],
    },
    {
        "name": "API Fetch Function",
        "instruction": "Write a JavaScript async function to fetch data from an API",
        "expected_keywords": ["try", "catch", "response.ok", "await", "fetch"],
    },
    {
        "name": "Division Function",
        "instruction": "Write a Python function to divide two numbers",
        "expected_keywords": ["try", "except", "ZeroDivisionError", "zero"],
    },
    {
        "name": "Database Query",
        "instruction": "Write a Python function to query a SQLite database",
        "expected_keywords": ["try", "except", "finally", "close", "?"],  # ? for parameterized
    },
]


def run_tests(agent: CoderAgentV3):
    """Run all tests and report results."""
    results = {
        "security": {"passed": 0, "failed": 0, "details": []},
        "error_handling": {"passed": 0, "failed": 0, "details": []},
    }

    print("\n" + "=" * 70)
    print("SECURITY TESTS")
    print("=" * 70)

    for test in SECURITY_TESTS:
        print(f"\n[TEST] {test['name']}")
        print("-" * 50)

        response = agent.generate(test["instruction"], test.get("code", ""))
        response_lower = response.lower()

        # Check for expected keywords
        found_keywords = [kw for kw in test["expected_keywords"] if kw.lower() in response_lower]
        passed = len(found_keywords) >= 2  # At least 2 keywords found

        status = "PASS" if passed else "FAIL"
        print(f"Response: {response[:200]}...")
        print(f"Found keywords: {found_keywords}")
        print(f"Result: {status}")

        if passed:
            results["security"]["passed"] += 1
        else:
            results["security"]["failed"] += 1

        results["security"]["details"].append({
            "name": test["name"],
            "passed": passed,
            "found_keywords": found_keywords,
        })

    print("\n" + "=" * 70)
    print("ERROR HANDLING TESTS")
    print("=" * 70)

    for test in ERROR_HANDLING_TESTS:
        print(f"\n[TEST] {test['name']}")
        print("-" * 50)

        response = agent.generate(test["instruction"])
        response_lower = response.lower()

        # Check for expected keywords
        found_keywords = [kw for kw in test["expected_keywords"] if kw.lower() in response_lower]
        has_try_catch = "try" in response_lower and ("except" in response_lower or "catch" in response_lower)
        passed = has_try_catch and len(found_keywords) >= 2

        status = "PASS" if passed else "FAIL"
        print(f"Response: {response[:200]}...")
        print(f"Has try-catch: {has_try_catch}")
        print(f"Found keywords: {found_keywords}")
        print(f"Result: {status}")

        if passed:
            results["error_handling"]["passed"] += 1
        else:
            results["error_handling"]["failed"] += 1

        results["error_handling"]["details"].append({
            "name": test["name"],
            "passed": passed,
            "has_try_catch": has_try_catch,
            "found_keywords": found_keywords,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    sec_total = results["security"]["passed"] + results["security"]["failed"]
    sec_pct = results["security"]["passed"] / sec_total * 100 if sec_total > 0 else 0

    err_total = results["error_handling"]["passed"] + results["error_handling"]["failed"]
    err_pct = results["error_handling"]["passed"] / err_total * 100 if err_total > 0 else 0

    print(f"\nSecurity Tests:      {results['security']['passed']}/{sec_total} ({sec_pct:.0f}%)")
    print(f"Error Handling:      {results['error_handling']['passed']}/{err_total} ({err_pct:.0f}%)")
    print(f"\nOverall: {results['security']['passed'] + results['error_handling']['passed']}/{sec_total + err_total}")

    return results


def main():
    print("Loading model...")
    agent = CoderAgentV3()
    agent.load()

    results = run_tests(agent)

    # Save results
    import json
    with open("evaluation/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to evaluation/results.json")


if __name__ == "__main__":
    main()
