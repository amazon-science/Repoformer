import os
import json
import argparse

LINE_API_REPOS = [
    "alibaba_FederatedScope",
    "google_vizier",
    "huggingface_evaluate",
    "opendilab_ACE",
    "awslabs_fortuna",
    "huggingface_diffusers",
    "pytorch_rl",
    "nerfstudio-project_nerfstudio"
]

FUNCTION_REPOS = [
    "amazon-science_patchcore-inspection",
    "deepmind_tracr",
    "google_lightweight_mmm",
    "lucidrains_imagen-pytorch",
    "CarperAI_trlx",
    "facebookresearch_omnivore",
    "leopard-ai_betty",
    "maxhumber_redframes"
]

repo_name_convert = {
    "CarperAI--trlx": "CarperAI_trlx",
    "lucidrains--imagen-pytorch": "lucidrains_imagen-pytorch",
    "deepmind--tracr": "deepmind_tracr",
    "leopard-ai--betty": "leopard-ai_betty",
    "google--lightweight_mmm": "google_lightweight_mmm",
    "amazon-science--patchcore-inspection": "amazon-science_patchcore-inspection",
    "facebookresearch--omnivore": "facebookresearch_omnivore",
    "maxhumber--redframes": "maxhumber_redframes"
}

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


def create_test_samples(triple):
    samples = []
    repo_dir, repo_type, prompt_file = triple
    # each line in the prompt file has the following format
    # {
    #     "prompt": "...",
    #     "metadata": {
    #         "task_id": "huggingface_diffusers/0",
    #         "ground_truth": "    StableDiffusionInpaintPipelineLegacy,",
    #         "fpath_tuple": [
    #             "huggingface_diffusers",
    #             "tests",
    #             "pipelines",
    #             "stable_diffusion",
    #             "test_stable_diffusion_inpaint_legacy.py"
    #         ],
    #         "context_start_lineno": 0,
    #         "line_no": 28}
    # }
    with open(prompt_file, 'r') as f:
        lines = f.readlines()
        prompts = [json.loads(l) for l in lines]

    repo_wise_ex_count = {}
    for p in prompts:
        s = {}
        task_id = p["metadata"]["task_id"]
        ground_truth = p["metadata"]["ground_truth"]
        num_gt_lines = len(ground_truth.split("\n"))

        repo_name = task_id.split("/")[0]
        repo_name = repo_name_convert.get(repo_name, repo_name)

        if repo_type == "function":
            # fix task_id
            if repo_name not in repo_wise_ex_count:
                repo_wise_ex_count[repo_name] = 0
            task_id = f"{repo_name}/{repo_wise_ex_count[repo_name]}"
            p["metadata"]["task_id"] = task_id
            repo_wise_ex_count[repo_name] += 1
            assert not ground_truth.startswith("\n")
            assert ground_truth.endswith("\n")
            num_gt_lines = num_gt_lines - 1
        else:
            assert not ground_truth.startswith("\n")
            assert not ground_truth.endswith("\n")

        assert repo_name == p["metadata"]["fpath_tuple"][0], \
            f"{repo_name} != {p['metadata']['fpath_tuple'][0]}"
        # fpath_tuple: ['huggingface_diffusers', 'src', 'diffusers', 'utils', 'doc_utils.py']
        # repo_name: huggingface_diffusers
        # filepath: huggingface_diffusers/src/diffusers/utils/doc_utils.py
        filepath = f"{os.sep}".join(p["metadata"]["fpath_tuple"])

        line_no_key = "line_no" if "line_no" in p["metadata"] else "lineno"
        if "context_start_lineno" not in p["metadata"]:
            continue
        if line_no_key not in p["metadata"]:
            continue

        # we need right context
        abs_filepath = os.path.join(repo_dir, filepath)
        if not os.path.exists(abs_filepath):
            raise FileNotFoundError(abs_filepath)

        with open(abs_filepath, "r") as tmp_file_in:
            file_content = tmp_file_in.read()
            lines_in_file = file_content.split("\n")

        prompt_start_line_no = p["metadata"]["context_start_lineno"]
        gt_start_line_no = p["metadata"][line_no_key]

        indent_len = len(lines_in_file[gt_start_line_no]) - len(lines_in_file[gt_start_line_no].lstrip())
        prefix = lines_in_file[gt_start_line_no][:indent_len]
        new_prompt = "\n".join(lines_in_file[:gt_start_line_no])
        new_prompt = new_prompt + "\n" + prefix

        new_gt_lines = []
        for idx in range(gt_start_line_no, gt_start_line_no + num_gt_lines):
            if idx == gt_start_line_no:
                new_gt_lines.append(lines_in_file[idx][indent_len:])
            else:
                new_gt_lines.append(lines_in_file[idx])
        new_ground_truth = "\n".join(new_gt_lines)

        rc_start_line_no = gt_start_line_no + num_gt_lines
        right_context = ""
        if rc_start_line_no < len(lines_in_file):
            right_context = "\n".join(lines_in_file[rc_start_line_no:rc_start_line_no + 50])
            right_context_full = "\n".join(lines_in_file[rc_start_line_no:])
            if repo_type == "line" or repo_type == "function":
                right_context = "\n" + right_context
                right_context_full = "\n" + right_context_full

        if new_ground_truth not in ground_truth:
            print("==" * 20)
            print(file_content)
            print("--" * 20 + " OLD GROUND TRUTH " + "--" * 20)
            print(ground_truth + "<CURSOR>")
            print("--" * 20 + " NEW GROUND TRUTH " + "--" * 20)
            print(new_ground_truth + "<eng_of_ground_truth>")
            raise ValueError("mismatch between ground_truth and new_ground_truth")
        if new_prompt + new_ground_truth + right_context not in file_content:
            print("==" * 20)
            print(file_content)
            print("--" * 20 + " PROMPT " + "--" * 20)
            print(new_prompt + "<eng_of_ground_truth>")
            print("--" * 20 + " GROUND TRUTH " + "--" * 20)
            print(new_ground_truth + "<eng_of_ground_truth>")
            print("--" * 20 + " RIGHT CONTEXT " + "--" * 20)
            print(right_context + "<eng_of_right_context>")
            raise ValueError("mismatch between {prompt, ground_truth, right_context} and file content")

        s["prompt"] = new_prompt
        s["groundtruth"] = new_ground_truth
        s["right_context"] = right_context
        s["full_right_context"] = right_context_full
        s["full_left_context"] = new_prompt
        p["metadata"].pop("ground_truth")
        p["metadata"].pop("context_start_lineno")
        p["metadata"].pop(line_no_key, None)
        p["metadata"]["filepath"] = filepath
        p["metadata"].pop("fpath_tuple")
        p["metadata"]["repository"] = URLS.get(repo_name, "")
        p["metadata"]["url"] = URLS.get(repo_name, "")
        s["metadata"] = p["metadata"]

        samples.append(s)

    return "\n".join([json.dumps(samp) for samp in samples])


def main():
    with open(args.output_file, 'w') as f:
        f.write(create_test_samples((args.repo_dir, args.repo_type, args.prompt_file)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to create data split")
    parser.add_argument("--repo_dir", required=True, help="repo root dir")
    parser.add_argument("--prompt_file", required=True, help="the folder of project context json files (absolute path)")
    parser.add_argument("--output_file", required=True, help="the output json file of model input")
    parser.add_argument("--repo_type", required=True, help="repo type: line or function")
    args = parser.parse_args()
    main()
