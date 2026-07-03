# Third-party licenses

brain-llm is licensed under the MIT License (see [LICENSE.MD](LICENSE.MD), Copyright (c) 2026 Alexandru Donea).
It builds on the third-party components listed below. Every one is permissively licensed and compatible with
MIT: all are attribution-only, with no copyleft.

This file exists so the project is covered when it redistributes or depends on those components.

---

## 1. Redistributed in this repository

One third-party file is vendored into this repository for offline use, so the license that covers it is
reproduced in full here.

### WordLlama

- File in this repo: `models/wordllama/tokenizers/l2_supercat_tokenizer_config.json`
- License: MIT
- Project: https://github.com/dleemiller/WordLlama
- Copyright (c) 2024 Lee Miller

```
MIT License

Copyright (c) 2024 Lee Miller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The `wordllama` Python package itself (which provides the embedding model used by `src/semantic.py`) is
installed via pip under the same MIT license shown above.

---

## 2. Runtime dependencies (installed via pip, not bundled here)

These are pulled onto your machine by `pip install -r requirements.txt`. Their source and license texts ship
inside the installed packages; this repository does not redistribute their code. They are listed for full
transparency.

### PyYAML

- Required. Reads and writes the on-disk memory stores (`src/runtime.py`).
- License: MIT
- Project: https://github.com/yaml/pyyaml
- Copyright (c) 2017-2021 Ingy döt Net; Copyright (c) 2006-2016 Kirill Simonov

```
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### numpy

- Required transitively (pulled in by `wordllama`; numpy-only inference, no PyTorch).
- License: BSD-3-Clause
- Project: https://github.com/numpy/numpy
- Copyright (c) 2005-2023, NumPy Developers. All rights reserved.

The full BSD-3-Clause text ships with the installed package (`numpy-*.dist-info/LICENSE.txt`). BSD-3-Clause
permits use, modification, and redistribution provided the copyright notice, the conditions, and the
disclaimer are retained, and the NumPy name is not used to endorse derived products without permission.

## Summary

| Component | Where | License | Bundled here |
|-----------|-------|---------|--------------|
| brain-llm | this repo | MIT | yes (this project) |
| WordLlama | `models/wordllama/` + pip | MIT | one tokenizer config file |
| PyYAML | pip | MIT | no |
| numpy | pip (via wordllama) | BSD-3-Clause | no |

All the dependency project licenses above are permissive and impose no copyleft; the only obligation is to keep
the copyright and license notices, which this file preserves.

Two clarifications, for completeness:

- **Other transitive dependencies.** Installing `wordllama` also pulls a few more packages (pydantic, requests,
  safetensors, tokenizers, toml, huggingface-hub, and their sub-deps), all under permissive MIT or Apache-2.0
  licenses. None is copyleft and none is redistributed in this repository, so there is no obligation to
  reproduce their notices; they are listed here only for transparency.
- **Compiler runtimes inside binary wheels.** The "no copyleft" statement refers to each dependency's own
  project license. Upstream binary wheels (notably numpy) bundle compiler runtime libraries (libgfortran,
  libquadmath) under the GCC Runtime Library Exception and LGPL. Those are dynamically-linked system runtimes
  installed via pip, not part of brain-llm's own distribution, and the GCC Runtime Library Exception explicitly
  permits their use in non-GPL programs.
