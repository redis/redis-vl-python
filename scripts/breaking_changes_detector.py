#!/usr/bin/env python3
"""
Breaking Changes Detector for RedisVL

Detects breaking changes between two versions by:
1. Analyzing public API exports (comparing __init__.py files)
2. Fetching and parsing GitHub release notes
3. Running old tests against new code with iterative patching
4. Generating comprehensive, actionable migration guides

Usage:
    python scripts/breaking_changes_detector.py <old_tag> <new_tag> [--output <file>]
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package is required. Install with: pip install openai")
    sys.exit(1)


@dataclass
class APIChange:
    """Represents a change in the public API."""

    change_type: str  # added, removed, moved, renamed
    symbol: str
    old_location: Optional[str]
    new_location: Optional[str]
    severity: str  # high, medium, low
    description: str


@dataclass
class BreakingChangesResult:
    """Result of the breaking changes detection."""

    old_tag: str
    new_tag: str
    api_changes: list = field(default_factory=list)
    release_notes_changes: list = field(default_factory=list)
    test_based_changes: list = field(default_factory=list)
    stage1_success: bool = False
    stage1_errors: Optional[str] = None
    stage2_success: Optional[bool] = None
    stage2_output: Optional[str] = None
    patches_applied: list = field(default_factory=list)
    patch_iterations: int = 0
    timestamp: str = ""


class BreakingChangesDetector:
    """Detects breaking changes using multiple analysis methods."""

    MAX_PATCH_ITERATIONS = 10
    MAX_UNIQUE_PATCHES = 50  # Prevent infinite loops

    # Key modules to analyze for public API
    PUBLIC_API_MODULES = [
        "redisvl/__init__.py",
        "redisvl/index/__init__.py",
        "redisvl/query/__init__.py",
        "redisvl/schema/__init__.py",
        "redisvl/extensions/__init__.py",
        "redisvl/extensions/llmcache/__init__.py",
        "redisvl/extensions/cache/__init__.py",
        "redisvl/extensions/router/__init__.py",
        "redisvl/extensions/session_manager/__init__.py",
        "redisvl/extensions/message_history/__init__.py",
        "redisvl/utils/vectorize/__init__.py",
        "redisvl/utils/rerank/__init__.py",
    ]

    def __init__(self, repo_path: str, model: str = "gpt-4o"):
        self.repo_path = Path(repo_path).resolve()
        self.model = model
        self.client = OpenAI()
        self._applied_patch_hashes = set()  # Track patches to avoid cycles

    def _run_command(
        self, cmd: list[str], cwd: Path, capture: bool = True, timeout: int = 300
    ) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        try:
            return subprocess.run(
                cmd, cwd=cwd, capture_output=capture, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(cmd, 1, "", "Command timed out")

    def _validate_tags(self, old_tag: str, new_tag: str) -> None:
        """Validate that both tags exist."""
        for tag in [old_tag, new_tag]:
            result = self._run_command(
                ["git", "rev-parse", "--verify", tag], self.repo_path
            )
            if result.returncode != 0:
                raise ValueError(f"Tag '{tag}' does not exist")

    # =========================================================================
    # PUBLIC API ANALYSIS
    # =========================================================================

    def _analyze_public_api(self, old_tag: str, new_tag: str) -> list[APIChange]:
        """Analyze changes in public API exports between versions."""
        print("\n=== Analyzing Public API Exports ===")
        changes = []

        for module_path in self.PUBLIC_API_MODULES:
            old_exports = self._get_module_exports(old_tag, module_path)
            new_exports = self._get_module_exports(new_tag, module_path)

            module_name = module_path.replace("/__init__.py", "").replace("/", ".")

            # Find removed exports
            for symbol in old_exports - new_exports:
                # Check if it moved somewhere else
                new_location = self._find_symbol_location(new_tag, symbol)
                if new_location and new_location != module_name:
                    changes.append(
                        APIChange(
                            change_type="moved",
                            symbol=symbol,
                            old_location=module_name,
                            new_location=new_location,
                            severity="high",
                            description=f"`{symbol}` moved from `{module_name}` to `{new_location}`",
                        )
                    )
                    print(f"  MOVED: {symbol}: {module_name} -> {new_location}")
                else:
                    changes.append(
                        APIChange(
                            change_type="removed",
                            symbol=symbol,
                            old_location=module_name,
                            new_location=None,
                            severity="high",
                            description=f"`{symbol}` removed from `{module_name}`",
                        )
                    )
                    print(f"  REMOVED: {symbol} from {module_name}")

            # Find added exports
            for symbol in new_exports - old_exports:
                changes.append(
                    APIChange(
                        change_type="added",
                        symbol=symbol,
                        old_location=None,
                        new_location=module_name,
                        severity="low",
                        description=f"`{symbol}` added to `{module_name}`",
                    )
                )

        print(
            f"  Found {len([c for c in changes if c.change_type in ('removed', 'moved')])} breaking API changes"
        )
        return changes

    def _get_module_exports(self, tag: str, module_path: str) -> set:
        """Get exported symbols from a module at a specific tag."""
        result = self._run_command(
            ["git", "show", f"{tag}:{module_path}"], self.repo_path
        )
        if result.returncode != 0:
            return set()

        content = result.stdout
        exports = set()

        # Parse __all__ if present
        all_match = re.search(r"__all__\s*=\s*\[(.*?)\]", content, re.DOTALL)
        if all_match:
            items = re.findall(r'["\'](\w+)["\']', all_match.group(1))
            exports.update(items)

        # Parse direct imports that are likely exports
        for match in re.finditer(
            r"from\s+[\w.]+\s+import\s+([^#\n(]+|\([^)]+\))", content
        ):
            items_str = match.group(1)
            items_str = re.sub(r"[(),]", " ", items_str)
            for item in items_str.split():
                item = item.strip()
                if item and item.isidentifier() and not item.startswith("_"):
                    exports.add(item)

        return exports

    def _find_symbol_location(self, tag: str, symbol: str) -> Optional[str]:
        """Find where a symbol is exported in a version."""
        for module_path in self.PUBLIC_API_MODULES:
            exports = self._get_module_exports(tag, module_path)
            if symbol in exports:
                return module_path.replace("/__init__.py", "").replace("/", ".")

        # Search in source files
        result = self._run_command(
            [
                "git",
                "grep",
                "-l",
                f"^class {symbol}\\|^def {symbol}\\|^{symbol} =",
                tag,
                "--",
                "redisvl/",
            ],
            self.repo_path,
        )
        if result.returncode == 0 and result.stdout.strip():
            first_file = result.stdout.strip().split("\n")[0]
            # Remove tag prefix if present
            if ":" in first_file:
                first_file = first_file.split(":", 1)[1]
            return first_file.replace("/", ".").replace(".py", "")

        return None

    # =========================================================================
    # RELEASE NOTES ANALYSIS
    # =========================================================================

    def _fetch_release_notes(self, old_tag: str, new_tag: str) -> list[dict]:
        """Fetch and parse release notes from GitHub."""
        print("\n=== Fetching Release Notes ===")
        changes = []

        # Normalize tags for comparison
        old_version = old_tag.lstrip("v")
        new_version = new_tag.lstrip("v")

        try:
            # Fetch releases from GitHub API
            url = "https://api.github.com/repos/redis/redis-vl-python/releases?per_page=100"
            with urlopen(url, timeout=30) as response:
                releases = json.loads(response.read().decode())

            for release in releases:
                tag = release.get("tag_name", "").lstrip("v")
                body = release.get("body", "") or ""

                # Check if this release is in our range
                if self._version_in_range(tag, old_version, new_version):
                    # Look for breaking changes in the release notes
                    breaking_patterns = [
                        r"breaking\s*change",
                        r"⚠️",
                        r"BREAKING",
                        r"renamed?\s+(?:from|to)",
                        r"removed?",
                        r"deprecated?",
                        r"migration",
                    ]

                    for pattern in breaking_patterns:
                        if re.search(pattern, body, re.IGNORECASE):
                            changes.append(
                                {
                                    "version": release.get("tag_name"),
                                    "title": release.get("name", ""),
                                    "body": body[:2000],  # Limit size
                                }
                            )
                            print(
                                f"  Found potential breaking changes in {release.get('tag_name')}"
                            )
                            break

        except (URLError, json.JSONDecodeError) as e:
            print(f"  Warning: Could not fetch release notes: {e}")

        return changes

    def _version_in_range(self, version: str, old: str, new: str) -> bool:
        """Check if a version is between old and new (exclusive of old, inclusive of new)."""
        try:

            def parse_version(v):
                return tuple(int(x) for x in re.findall(r"\d+", v)[:3])

            v = parse_version(version)
            o = parse_version(old)
            n = parse_version(new)
            return o < v <= n
        except (ValueError, IndexError):
            return False

    def _analyze_release_notes_with_llm(
        self, release_notes: list[dict], old_tag: str, new_tag: str
    ) -> list[dict]:
        """Use LLM to extract breaking changes from release notes."""
        if not release_notes:
            return []

        print("  Analyzing release notes with LLM...")

        notes_text = "\n\n---\n\n".join(
            [f"## {r['version']}: {r['title']}\n{r['body']}" for r in release_notes]
        )

        prompt = f"""Extract ALL breaking changes from these release notes for versions between {old_tag} and {new_tag}.

Release Notes:
{notes_text[:15000]}

For each breaking change, provide:
1. The version it was introduced
2. What changed (be specific with class/function names)
3. Old code example
4. New code example
5. Severity (high/medium/low)

Return as JSON array:
```json
[
  {{
    "version": "v0.11.0",
    "change": "HybridQuery renamed to AggregateHybridQuery",
    "old_code": "from redisvl.query.aggregate import HybridQuery",
    "new_code": "from redisvl.query.aggregate import AggregateHybridQuery",
    "severity": "high"
  }}
]
```

Only include ACTUAL breaking changes, not new features.
Return ONLY valid JSON."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract breaking changes from release notes. Be specific.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2000,
        )

        content = response.choices[0].message.content
        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                json_str = content
            return json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError):
            return []

    # =========================================================================
    # TEST-BASED ANALYSIS
    # =========================================================================

    def _setup_test_environment(
        self, work_dir: Path, old_tag: str, new_tag: str
    ) -> Path:
        """Set up test environment with new code and old tests."""
        print("\n=== Setting Up Test Environment ===")

        print("  Cloning repository...")
        self._run_command(
            ["git", "clone", "--quiet", str(self.repo_path), str(work_dir / "repo")],
            work_dir,
        )
        repo_dir = work_dir / "repo"

        print(f"  Checking out {new_tag}...")
        self._run_command(["git", "checkout", "--quiet", new_tag], repo_dir)

        print(f"  Extracting tests from {old_tag}...")
        old_tests_dir = work_dir / "old_tests"
        old_tests_dir.mkdir()

        archive_result = subprocess.run(
            ["git", "archive", old_tag, "tests"],
            cwd=self.repo_path,
            capture_output=True,
        )
        if archive_result.returncode != 0:
            raise RuntimeError(f"Failed to archive tests: {archive_result.stderr}")

        tar_path = work_dir / "old_tests.tar"
        tar_path.write_bytes(archive_result.stdout)
        subprocess.run(
            ["tar", "-xf", str(tar_path), "-C", str(old_tests_dir)], check=True
        )

        print("  Replacing tests directory...")
        shutil.rmtree(repo_dir / "tests")
        shutil.copytree(old_tests_dir / "tests", repo_dir / "tests")

        print("  Installing dependencies...")
        self._run_command(["uv", "sync", "--all-extras", "--quiet"], repo_dir)

        return repo_dir

    def _run_test_collection(self, repo_dir: Path) -> tuple[bool, str, list]:
        """Try to collect tests and capture import errors."""
        result = self._run_command(
            [
                "uv",
                "run",
                "python",
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "--tb=short",
                "tests/",
            ],
            repo_dir,
            timeout=120,
        )

        output = result.stdout + "\n" + result.stderr
        errors = []

        # Parse import errors
        for match in re.finditer(
            r"ImportError[^:]*:\s*cannot import name ['\"](\w+)['\"] from ['\"]([^'\"]+)['\"]",
            output,
        ):
            errors.append(
                {
                    "type": "import_error",
                    "symbol": match.group(1),
                    "module": match.group(2),
                }
            )

        for match in re.finditer(
            r"ModuleNotFoundError.*No module named ['\"]([^'\"]+)['\"]", output
        ):
            errors.append(
                {
                    "type": "module_not_found",
                    "module": match.group(1),
                }
            )

        # Deduplicate
        seen = set()
        unique_errors = []
        for e in errors:
            key = json.dumps(e, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_errors.append(e)

        return result.returncode == 0, output, unique_errors

    def _run_tests(self, repo_dir: Path) -> tuple[bool, str]:
        """Run tests and return results."""
        result = self._run_command(
            ["uv", "run", "python", "-m", "pytest", "-v", "--tb=short", "-x", "tests/"],
            repo_dir,
            timeout=600,
        )
        return result.returncode == 0, result.stdout + "\n" + result.stderr

    def _get_failing_files_content(self, repo_dir: Path, output: str) -> dict:
        """Get content of failing test files."""
        files = set()
        for match in re.finditer(r"(tests/\S+\.py):\d+:", output):
            files.add(match.group(1))
        for match in re.finditer(r"ERROR collecting (tests/\S+\.py)", output):
            files.add(match.group(1))

        contents = {}
        for f in list(files)[:10]:  # Limit to 10 files
            file_path = repo_dir / f
            if file_path.exists():
                content = file_path.read_text()
                # Get imports section
                lines = content.split("\n")[:60]
                contents[f] = "\n".join(lines)

        return contents

    def _generate_patches(
        self,
        error_output: str,
        parsed_errors: list,
        api_changes: list[APIChange],
        repo_dir: Path,
        old_tag: str,
        new_tag: str,
        iteration: int,
    ) -> list[dict]:
        """Generate patches based on errors and API analysis."""
        print(f"\n=== Generating Patches (Iteration {iteration}) ===")

        # Get failing file contents
        file_contents = self._get_failing_files_content(repo_dir, error_output)

        # Build API changes context
        api_context = []
        for change in api_changes:
            if change.change_type in ("moved", "removed"):
                api_context.append(
                    {
                        "symbol": change.symbol,
                        "old": change.old_location,
                        "new": change.new_location,
                        "type": change.change_type,
                    }
                )

        prompt = f"""Fix Python import errors to make old tests ({old_tag}) work with new code ({new_tag}).

## Import Errors:
{json.dumps(parsed_errors, indent=2)}

## Known API Changes (symbols that moved or were removed):
{json.dumps(api_context, indent=2)}

## Failing Test File Contents (imports section):
{json.dumps(file_contents, indent=2)}

## Error Output:
```
{error_output[:10000]}
```

Generate patches to fix these imports. For each error:
1. Find the EXACT import line in the file content provided
2. Create the corrected import based on API changes
3. Include the FULL import statement, not partial

Return as JSON array:
```json
[
  {{
    "file": "tests/unit/test_something.py",
    "old_code": "from redisvl.query.aggregate import HybridQuery",
    "new_code": "from redisvl.query.aggregate import AggregateHybridQuery",
    "explanation": "HybridQuery was renamed to AggregateHybridQuery"
  }}
]
```

CRITICAL:
- old_code must EXACTLY match text in the file
- For multi-line imports, include the full import block
- Return ONLY valid JSON"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Generate exact code patches. Match old_code exactly to file contents.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4000,
        )

        content = response.choices[0].message.content
        patches = []

        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                json_str = content
            patches = json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError) as e:
            print(f"  Warning: Could not parse patches: {e}")

        # Filter out already-tried patches
        new_patches = []
        for patch in patches:
            patch_hash = hashlib.md5(
                f"{patch.get('file', '')}:{patch.get('old_code', '')}".encode()
            ).hexdigest()
            if patch_hash not in self._applied_patch_hashes:
                new_patches.append(patch)

        print(
            f"  Generated {len(new_patches)} new patches (filtered {len(patches) - len(new_patches)} duplicates)"
        )
        return new_patches

    def _apply_patches(self, repo_dir: Path, patches: list[dict]) -> list[dict]:
        """Apply patches and return successfully applied ones."""
        applied = []

        for patch in patches:
            try:
                file_path = repo_dir / patch.get("file", "")
                if not file_path.exists():
                    continue

                old_code = patch.get("old_code", "")
                new_code = patch.get("new_code", "")

                if not old_code or old_code == new_code:
                    continue

                # Create hash to track this patch
                patch_hash = hashlib.md5(
                    f"{patch.get('file', '')}:{old_code}".encode()
                ).hexdigest()

                if patch_hash in self._applied_patch_hashes:
                    continue

                content = file_path.read_text()
                if old_code in content:
                    content = content.replace(old_code, new_code, 1)
                    file_path.write_text(content)
                    self._applied_patch_hashes.add(patch_hash)
                    applied.append(patch)
                    print(
                        f"  Applied: {patch.get('file')} - {patch.get('explanation', '')[:50]}"
                    )

            except Exception as e:
                print(f"  Error: {e}")

        return applied

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def _generate_final_report(
        self, result: BreakingChangesResult, old_tag: str, new_tag: str
    ) -> str:
        """Generate comprehensive report combining all sources."""
        print("\n=== Generating Final Report ===")

        # Compile all breaking changes
        all_changes = []

        # From API analysis
        for change in result.api_changes:
            if change.change_type in ("removed", "moved"):
                all_changes.append(
                    {
                        "source": "API Analysis",
                        "type": change.change_type,
                        "symbol": change.symbol,
                        "old": change.old_location,
                        "new": change.new_location,
                        "severity": change.severity,
                    }
                )

        # From release notes
        for change in result.release_notes_changes:
            all_changes.append(
                {
                    "source": f"Release Notes ({change.get('version', 'unknown')})",
                    "type": "documented",
                    "description": change.get("change", ""),
                    "old_code": change.get("old_code", ""),
                    "new_code": change.get("new_code", ""),
                    "severity": change.get("severity", "high"),
                }
            )

        # From patches applied
        for patch in result.patches_applied:
            all_changes.append(
                {
                    "source": "Test Analysis",
                    "type": "import_change",
                    "file": patch.get("file", ""),
                    "old_code": patch.get("old_code", ""),
                    "new_code": patch.get("new_code", ""),
                    "explanation": patch.get("explanation", ""),
                }
            )

        prompt = f"""Generate a comprehensive breaking changes report for upgrading from {old_tag} to {new_tag}.

## All Detected Changes:
{json.dumps(all_changes, indent=2)}

## Stage 1 (Import) Status: {"PASSED" if result.stage1_success else "FAILED"}
## Stage 2 (Tests) Status: {"PASSED" if result.stage2_success else "FAILED" if result.stage2_success is not None else "Not Run"}

## Test Output (if failed):
```
{result.stage2_output[:5000] if result.stage2_output and not result.stage2_success else "N/A"}
```

Generate a report with these sections:

1. **Executive Summary** - Brief overview for decision makers

2. **Breaking Changes** - Table format:
   | Change | Old | New | Severity | Action Required |

3. **Migration Guide** - For each breaking change:
   - What changed
   - Old code example
   - New code example
   - Search pattern to find affected code

4. **Automated Migration Commands** - sed/grep commands

5. **New Features** (brief list of additions)

6. **Behavioral Changes** (from test failures if any)

Be SPECIFIC. Use exact class names, import paths, and code examples.
Do NOT say "check documentation" - provide the actual fix."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Create specific, actionable migration documentation.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=6000,
        )

        return response.choices[0].message.content

    # =========================================================================
    # MAIN DETECTION FLOW
    # =========================================================================

    def detect(self, old_tag: str, new_tag: str) -> BreakingChangesResult:
        """Detect breaking changes using all available methods."""
        print(f"\n{'='*60}")
        print(f"Breaking Changes Detector")
        print(f"Comparing: {old_tag} -> {new_tag}")
        print(f"{'='*60}")

        self._validate_tags(old_tag, new_tag)
        self._applied_patch_hashes.clear()

        result = BreakingChangesResult(
            old_tag=old_tag,
            new_tag=new_tag,
            timestamp=datetime.now().isoformat(),
        )

        # Method 1: Analyze public API exports
        result.api_changes = self._analyze_public_api(old_tag, new_tag)

        # Method 2: Fetch and analyze release notes
        release_notes = self._fetch_release_notes(old_tag, new_tag)
        result.release_notes_changes = self._analyze_release_notes_with_llm(
            release_notes, old_tag, new_tag
        )

        # Method 3: Test-based analysis
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)

            try:
                repo_dir = self._setup_test_environment(work_dir, old_tag, new_tag)

                all_patches = []
                iteration = 0

                while iteration < self.MAX_PATCH_ITERATIONS:
                    if len(self._applied_patch_hashes) >= self.MAX_UNIQUE_PATCHES:
                        print(
                            f"  Reached max unique patches ({self.MAX_UNIQUE_PATCHES})"
                        )
                        break

                    iteration += 1
                    print(
                        f"\n=== Test Collection Attempt {iteration}/{self.MAX_PATCH_ITERATIONS} ==="
                    )

                    success, output, errors = self._run_test_collection(repo_dir)

                    if success:
                        print("Stage 1 PASSED")
                        result.stage1_success = True
                        result.stage1_errors = output
                        break

                    result.stage1_errors = output
                    print(f"Stage 1 FAILED: {len(errors)} unique errors")

                    if not errors:
                        print("  No parseable errors, stopping")
                        break

                    patches = self._generate_patches(
                        output,
                        errors,
                        result.api_changes,
                        repo_dir,
                        old_tag,
                        new_tag,
                        iteration,
                    )

                    if not patches:
                        print("  No new patches to apply")
                        break

                    applied = self._apply_patches(repo_dir, patches)
                    if not applied:
                        print("  No patches could be applied")
                        break

                    all_patches.extend(applied)
                    result.patches_applied = all_patches

                result.patch_iterations = iteration

                # Stage 2: Run tests
                if result.stage1_success:
                    print("\n=== Stage 2: Running Tests ===")
                    success, output = self._run_tests(repo_dir)
                    result.stage2_success = success
                    result.stage2_output = output
                    print(f"Stage 2 {'PASSED' if success else 'FAILED'}")

            except Exception as e:
                print(f"\nError: {e}")
                import traceback

                traceback.print_exc()

        return result


def format_report(result: BreakingChangesResult, detailed_analysis: str) -> str:
    """Format the final markdown report."""
    report = []
    report.append(f"# Breaking Changes Report: {result.old_tag} → {result.new_tag}")
    report.append(f"\n**Generated:** {result.timestamp}")
    report.append("")

    # Quick stats
    api_breaking = len(
        [c for c in result.api_changes if c.change_type in ("removed", "moved")]
    )
    report.append("## Quick Stats")
    report.append("")
    report.append(f"- **API Breaking Changes:** {api_breaking}")
    report.append(f"- **Release Notes Changes:** {len(result.release_notes_changes)}")
    report.append(f"- **Test-based Patches:** {len(result.patches_applied)}")
    report.append(
        f"- **Stage 1 (Imports):** {'✅ PASSED' if result.stage1_success else '❌ FAILED'}"
    )
    if result.stage2_success is not None:
        report.append(
            f"- **Stage 2 (Tests):** {'✅ PASSED' if result.stage2_success else '❌ FAILED'}"
        )
    report.append("")

    # API Changes
    if result.api_changes:
        breaking = [
            c for c in result.api_changes if c.change_type in ("removed", "moved")
        ]
        if breaking:
            report.append("## API Changes (from export analysis)")
            report.append("")
            report.append("| Symbol | Change | Old Location | New Location |")
            report.append("|--------|--------|--------------|--------------|")
            for c in breaking:
                report.append(
                    f"| `{c.symbol}` | {c.change_type} | `{c.old_location or 'N/A'}` | `{c.new_location or 'REMOVED'}` |"
                )
            report.append("")

    # Release Notes Changes
    if result.release_notes_changes:
        report.append("## Documented Breaking Changes (from release notes)")
        report.append("")
        for c in result.release_notes_changes:
            report.append(f"### {c.get('version', 'Unknown')}: {c.get('change', '')}")
            report.append("")
            if c.get("old_code"):
                report.append("**Before:**")
                report.append(f"```python\n{c['old_code']}\n```")
            if c.get("new_code"):
                report.append("**After:**")
                report.append(f"```python\n{c['new_code']}\n```")
            report.append("")

    # Test-based patches
    if result.patches_applied:
        report.append("## Required Code Changes (from test analysis)")
        report.append("")
        # Deduplicate by explanation
        seen = set()
        for p in result.patches_applied:
            key = p.get("explanation", "")[:50]
            if key in seen:
                continue
            seen.add(key)
            report.append(f"### {p.get('explanation', 'Import fix')}")
            report.append("")
            report.append(f"**File:** `{p.get('file', '')}`")
            report.append("")
            report.append("```python")
            report.append(f"# Old:\n{p.get('old_code', '')}")
            report.append(f"\n# New:\n{p.get('new_code', '')}")
            report.append("```")
            report.append("")

    # Detailed analysis
    report.append("## Detailed Analysis")
    report.append("")
    report.append(detailed_analysis)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Detect breaking changes between versions"
    )
    parser.add_argument("old_tag", help="The older version tag")
    parser.add_argument("new_tag", help="The newer version tag")
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument("--model", "-m", default="gpt-4o", help="OpenAI model")
    parser.add_argument("--repo", "-r", default=".", help="Repository path")

    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable required")
        sys.exit(1)

    detector = BreakingChangesDetector(args.repo, model=args.model)
    result = detector.detect(args.old_tag, args.new_tag)

    detailed = detector._generate_final_report(result, args.old_tag, args.new_tag)
    report = format_report(result, detailed)

    if args.output:
        Path(args.output).write_text(report)
        print(f"\nReport saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print(report)


if __name__ == "__main__":
    main()
