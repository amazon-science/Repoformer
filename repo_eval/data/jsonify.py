"""
Script to compile the files under packages into the source code
jsonl file that CodeTransform package can digest
"""
import json
import os
import glob
import argparse
from typing import List
from pathlib import Path

REPOS = [
    "alibaba_FederatedScope",
    "google_vizier",
    "huggingface_evaluate",
    "opendilab_ACE",
    "awslabs_fortuna",
    "huggingface_diffusers",
    "pytorch_rl",
    "nerfstudio-project_nerfstudio",
    "amazon-science_patchcore-inspection",
    "deepmind_tracr",
    "google_lightweight_mmm",
    "lucidrains_imagen-pytorch",
    "CarperAI_trlx",
    "facebookresearch_omnivore",
    "leopard-ai_betty",
    "maxhumber_redframes"
]

URLS = {
    "alibaba_FederatedScope": "https://github.com/alibaba/FederatedScope",
    "google_vizier": "https://github.com/google/vizier",
    "huggingface_evaluate": "https://github.com/huggingface/evaluate",
    "opendilab_ACE": "https://github.com/opendilab/ACE",
    "awslabs_fortuna": "https://github.com/awslabs/fortuna",
    "huggingface_diffusers": "https://github.com/huggingface/diffusers",
    "pytorch_rl": "https://github.com/pytorch/rl",
    "nerfstudio-project_nerfstudio": "https://github.com/nerfstudio-project/nerfstudio",
    "amazon-science_patchcore-inspection": "https://github.com/amazon-science/patchcore-inspection",
    "deepmind_tracr": "https://github.com/deepmind/tracr",
    "google_lightweight_mmm": "https://github.com/google/lightweight_mmm",
    "lucidrains_imagen-pytorch": "https://github.com/lucidrains/imagen-pytorch",
    "CarperAI_trlx": "https://github.com/CarperAI/trlx",
    "facebookresearch_omnivore": "https://github.com/facebookresearch/omnivore",
    "leopard-ai_betty": "https://github.com/leopard-ai/betty",
    "maxhumber_redframes": "https://github.com/maxhumber/redframes"
}


def process_single_file(file_content: str):
    """
    Wrap a single file content
    """
    return {
        "type": "code",
        "lang": "python",
        "content": file_content
    }


def main(repo_dir, repo_name):
    source = "github"
    url = URLS.get(repo_name, "")
    license = "permissive"
    patent = "NO_PATENT_INFO"
    repository = URLS.get(repo_name, "")

    codetransform_jsons = []
    filelist = list(Path(os.path.join(repo_dir, repo_name)).rglob("*.py"))
    for file in filelist:
        file_content = open(file).read()
        content_json = process_single_file(file_content)
        filepath = os.path.relpath(file, repo_dir)
        composed_json = {
            "source": source,
            "url": url,
            "license": license,
            "patent": patent,
            "repository": repository,
            "filepath": filepath,
            "content": [content_json]
        }
        codetransform_jsons.append(composed_json)

    return codetransform_jsons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", type=str, required=True, help="directory, where subdir is pacakge dir")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir")
    args = parser.parse_args()

    with open(os.path.join(args.output_dir, f"repocoder_packages.jsonl"), "w") as out:
        for repo_name in REPOS:
            package_jsons = main(args.repo_dir, repo_name)
            for package_json in package_jsons:
                json.dump(package_json, out)
                out.write("\n")
