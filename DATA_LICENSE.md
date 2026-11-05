# Dataset license and attribution

The files under `dataset/` are a modified, stagewise redistribution of
PersonaMem-v2.

PersonaMem-v2 is licensed under the Creative Commons Attribution 4.0
International License (CC BY 4.0):

- Source: <https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2>
- License: <https://creativecommons.org/licenses/by/4.0/>
- Citation: Jiang et al., "PersonaMem-v2: Towards Personalized Intelligence via
  Learning Implicit User Personas and Agentic Memory," arXiv:2512.06688 (2025).

We redistribute the source histories and derived stage profiles under CC BY
4.0. Changes from the upstream data:

- removed system messages and empty messages;
- kept the first `m` messages under the 32K history budget;
- split the selected history into eight chronological stages;
- generated cumulative stage profiles with `gpt-5.6-sol` using `sol_fast_v1`.

The code in this repository is licensed separately under the MIT License; see
`LICENSE`. The PPL-Writer adapter is trained from Qwen3.5-4B (Apache-2.0); the
base model weights are not included in this repository.
