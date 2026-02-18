import requests
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Optional

REGISTRY_API      = "https://registry.terraform.io/v1/modules"
RAW_BASE          = "https://raw.githubusercontent.com"
TREE_API          = "https://api.github.com/repos"       
DEFAULT_BRANCHES  = ["main", "master"]
TARGET_FOLDERS    = {"examples", "wrappers"}
MIN_FILE_BYTES    = 50
MAX_FILE_BYTES    = 50_000
REQUEST_DELAY     = 0.4   

LICENSE_KEYWORD_MAP = {
    "apache license 2.0":          "apache-2.0",
    "apache license, version 2.0": "apache-2.0",
    "mit license":                 "mit",
    "mit":                         "mit",
    "bsd 2-clause":                "bsd-2-clause",
    "bsd 3-clause":                "bsd-3-clause",
    "mozilla public license 2.0":  "mpl-2.0",
    "mozilla public license":      "mpl-2.0",
    "isc license":                 "isc",
    "isc":                         "isc",
    "unlicense":                   "unlicense",
    "zero-clause bsd":             "0bsd",
}

ACCEPTED_LICENSES = set(LICENSE_KEYWORD_MAP.values())

session = requests.Session()
session.headers.update({"User-Agent": "TerraformDatasetBuilder/1.0"})


def get(url: str, timeout: int = 20) -> Optional[requests.Response]:
    """Safe GET with retry on transient errors."""
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == 2:
                print(f"      [!] Failed {url} — {e}")
            time.sleep(1)
    return None


def load_services_config(path: str = "services_config.json") -> Dict[str, List[str]]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        default = {
            "aws":     ["ec2", "s3", "rds", "lambda", "vpc"],
            "google":  ["compute", "storage", "sql", "gke"],
            "azurerm": ["virtual-machine", "storage-account", "aks"],
            "oci":     ["compute", "object-storage", "database"]
        }
        cfg_path.write_text(json.dumps(default, indent=2))
        print(f"[*] Created default services_config.json — edit it to customize.")

    config = json.loads(cfg_path.read_text())
    print("[*] Loaded services config:")
    for provider, services in config.items():
        print(f"    {provider}: {services}")
    return config


def fetch_registry_modules(provider: str, page_limit: int = 100) -> List[Dict]:
    """Paginate through the Terraform Registry for a given provider."""
    modules, offset = [], 0
    print(f"\n[*] Fetching registry modules — provider: {provider}")

    while True:
        url  = f"{REGISTRY_API}?provider={provider}&limit={page_limit}&offset={offset}"
        resp = get(url)
        if not resp:
            break
        data = resp.json()
        batch = data.get("modules", [])
        if not batch:
            break
        modules.extend(batch)
        print(f"    ... {len(modules)} modules retrieved")
        offset += page_limit
        time.sleep(REQUEST_DELAY)

    print(f"    Total: {len(modules)}")
    return modules


def filter_by_service(modules: List[Dict], services: List[str]) -> List[Dict]:
    """Keep only modules whose name contains a target service keyword."""
    matched = []
    for m in modules:
        name_lower = m.get("name", "").lower()
        for svc in services:
            if svc.lower() in name_lower:
                m["matched_service"] = svc
                matched.append(m)
                break
    print(f"    Filtered to {len(matched)} modules matching services: {services}")
    return matched


def get_module_source(namespace: str, name: str, provider: str) -> Optional[str]:
    """Return the source GitHub URL for a registry module."""
    url  = f"{REGISTRY_API}/{namespace}/{name}/{provider}"
    resp = get(url)
    return resp.json().get("source") if resp else None


def parse_github_owner_repo(source_url: str) -> Optional[tuple]:
    """Extract (owner, repo) from a GitHub source URL."""
    if not source_url or "github.com" not in source_url:
        return None
    m = re.search(r"github\.com[:/]([^/]+)/([^/.\s]+)", source_url)
    return (m.group(1), m.group(2)) if m else None


LICENSE_FILENAMES = ["LICENSE", "LICENSE.md", "LICENSE.txt", "license", "License"]

def detect_license(owner: str, repo: str, branch: str) -> Optional[str]:
    """
    Download the LICENSE file directly via raw URL and detect the SPDX ID
    by scanning its content for known keywords.
    No GitHub API token required.
    """
    for fname in LICENSE_FILENAMES:
        url  = f"{RAW_BASE}/{owner}/{repo}/{branch}/{fname}"
        resp = get(url)
        if not resp:
            continue

        content_lower = resp.text.lower()
        for keyword, spdx_id in LICENSE_KEYWORD_MAP.items():
            if keyword in content_lower:
                return spdx_id

        return "unknown"

    return None   


def get_default_branch(owner: str, repo: str) -> str:
    """
    Try to read the default branch from the public GitHub tree API.
    Falls back to trying 'main' then 'master' if unreachable.
    No auth needed for public repos.
    """
    url  = f"{TREE_API}/{owner}/{repo}"
    resp = get(url)
    if resp:
        branch = resp.json().get("default_branch")
        if branch:
            return branch

    for branch in DEFAULT_BRANCHES:
        probe = f"{RAW_BASE}/{owner}/{repo}/{branch}/README.md"
        r     = get(probe)
        if r:
            return branch

    return "main"

def get_repo_tree(owner: str, repo: str, branch: str) -> List[Dict]:
    """
    Fetch the full recursive file tree of a repo using GitHub's public
    tree API endpoint. This is a single request regardless of repo size
    and does NOT require authentication for public repos.
    """
    url  = f"{TREE_API}/{owner}/{repo}/git/trees/{branch}?recursive=1"
    resp = get(url)
    if not resp:
        return []
    data = resp.json()
    if data.get("truncated"):
        print(f"      [!] Tree truncated for {owner}/{repo} — large repo")
    return data.get("tree", [])


def filter_target_tf_files(tree: List[Dict]) -> List[str]:
    """
    From the full repo tree, return paths of .tf files that live inside
    an 'examples' or 'wrappers' folder at any depth.
    """
    target_paths = []
    for item in tree:
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        if not path.endswith(".tf"):
            continue
       
        parts = path.lower().split("/")
        if any(part in TARGET_FOLDERS for part in parts):
            target_paths.append(path)
    return target_paths

def download_tf_file(owner: str, repo: str, branch: str, path: str) -> Optional[str]:
    """Download a .tf file directly via raw.githubusercontent.com."""
    url  = f"{RAW_BASE}/{owner}/{repo}/{branch}/{path}"
    resp = get(url)
    if not resp:
        return None
    content = resp.text
    if MIN_FILE_BYTES <= len(content) <= MAX_FILE_BYTES:
        return content
    return None


def save_tf_file(content: str, output_dir: Path, provider: str,
                 service: str, module_id: str, filename: str):
    """Save a .tf file preserving provider/service/module hierarchy."""
    dest = output_dir / provider / service / module_id / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return dest


def process_module(module: Dict, provider: str, output_dir: Path) -> Dict:
    """Full pipeline for a single registry module."""
    ns      = module["namespace"]
    name    = module["name"]
    service = module["matched_service"]
    mod_id  = f"{ns}__{name}"

    result = {
        "namespace": ns, "name": name,
        "provider": provider, "service": service,
        "license": None, "files_saved": 0, "status": "skipped"
    }

    source = get_module_source(ns, name, provider)
    time.sleep(REQUEST_DELAY)
    if not source:
        result["status"] = "no_source"
        return result

    gh = parse_github_owner_repo(source)
    if not gh:
        result["status"] = "not_github"
        return result

    owner, repo = gh
    print(f"    → {owner}/{repo}")

    branch = get_default_branch(owner, repo)
    time.sleep(REQUEST_DELAY)

    license_id = detect_license(owner, repo, branch)
    result["license"] = license_id
    time.sleep(REQUEST_DELAY)

    if not license_id or license_id not in ACCEPTED_LICENSES:
        print(f"    ✗ License rejected: {license_id}")
        result["status"] = "license_rejected"
        return result

    print(f"    ✓ License accepted: {license_id}")

    tree = get_repo_tree(owner, repo, branch)
    time.sleep(REQUEST_DELAY)
    if not tree:
        result["status"] = "empty_tree"
        return result

    tf_paths = filter_target_tf_files(tree)
    if not tf_paths:
        print(f"    ✗ No .tf files in examples/wrappers")
        result["status"] = "no_target_files"
        return result

    print(f"    ✓ Found {len(tf_paths)} .tf files in target folders")

    for path in tf_paths:
        content = download_tf_file(owner, repo, branch, path)
        time.sleep(REQUEST_DELAY)
        if content:
            filename = path.replace("/", "__")   # flatten path into filename
            save_tf_file(content, output_dir, provider, service, mod_id, filename)
            result["files_saved"] += 1

    result["status"] = "success" if result["files_saved"] > 0 else "no_files_downloaded"
    print(f"    ✓ Saved {result['files_saved']} files")
    return result

def run(services_config_path: str = "services_config.json",
        output_dir: str = "terraform_training_data",
        max_modules_per_provider: Optional[int] = None):

    config     = load_services_config(services_config_path)
    out        = Path(output_dir)
    out.mkdir(exist_ok=True)
    all_stats  = {}

    for provider, services in config.items():
        print(f"\n{'='*60}")
        print(f"PROVIDER: {provider.upper()}")
        print(f"{'='*60}")

        all_modules = fetch_registry_modules(provider)

        modules = filter_by_service(all_modules, services)

        if max_modules_per_provider:
            modules = modules[:max_modules_per_provider]

        results   = []
        succeeded = 0

        for i, module in enumerate(modules, 1):
            print(f"\n  [{i}/{len(modules)}] {module['namespace']}/{module['name']}")
            res = process_module(module, provider, out)
            results.append(res)
            if res["status"] == "success":
                succeeded += 1

        all_stats[provider] = {
            "total": len(results),
            "succeeded": succeeded,
            "results": results
        }

        print(f"\n  Summary — {provider}: {succeeded}/{len(results)} modules scraped successfully")

    summary_path = out / "scrape_summary.json"
    summary_path.write_text(json.dumps(all_stats, indent=2))

    print(f"\n{'='*60}")
    print(f"✓ Scraping complete!")
    print(f"✓ Dataset saved to: {out.resolve()}")
    print(f"✓ Summary saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys

    if not hasattr(sys, '_scraper_running'):
        sys._scraper_running = True
        run(
            services_config_path="services_config.json",
            output_dir="terraform_training_data",
            max_modules_per_provider=5       

        )
