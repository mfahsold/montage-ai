# Third-Party Licenses and Attributions

**Montage AI** uses open source software components. This document lists all third-party dependencies, their licenses, and required attributions.

---

## Core Dependencies

### FFmpeg
- **License:** LGPL 2.1+ / GPL 2+ (depending on build configuration)
- **Website:** https://ffmpeg.org/
- **Purpose:** Video/audio encoding, decoding, and processing
- **Usage:** Core video processing pipeline
- **Note:** We use FFmpeg under LGPL. Our build does not link against GPL-only libraries.

```
FFmpeg is free software; you can redistribute it and/or modify it under 
the terms of the GNU Lesser General Public License as published by the 
Free Software Foundation; either version 2.1 of the License, or (at your 
option) any later version.
```

### OpenCV (opencv-python)
- **License:** Apache License 2.0
- **Website:** https://opencv.org/
- **Purpose:** Computer vision, scene detection, face detection
- **Usage:** Scene analysis, visual content detection

```
Copyright 2000-2024, OpenCV contributors
Licensed under the Apache License, Version 2.0
```

### Librosa
- **License:** ISC License
- **Website:** https://librosa.org/
- **Purpose:** Audio analysis, beat detection
- **Usage:** Music beat analysis, energy detection

```
Copyright 2013-2024, Brian McFee et al.
ISC License
```

### OpenAI Whisper
- **License:** MIT License
- **Website:** https://github.com/openai/whisper
- **Purpose:** Speech recognition, transcription
- **Usage:** Video transcription, caption generation

```
Copyright 2022 OpenAI
MIT License
```

### Demucs / HT Demucs
- **License:** MIT License
- **Website:** https://github.com/facebookresearch/demucs
- **Purpose:** Audio source separation, voice isolation
- **Usage:** Voice isolation from background audio

```
Copyright (c) Meta Platforms, Inc. and affiliates
MIT License
```

### Real-ESRGAN
- **License:** BSD 3-Clause License
- **Website:** https://github.com/xinntao/Real-ESRGAN
- **Purpose:** AI-based image/video upscaling
- **Usage:** 4x video upscaling

```
Copyright (c) 2021 Xintao Wang
BSD 3-Clause License
```

### OpenTimelineIO (OTIO)
- **License:** Apache License 2.0
- **Website:** https://github.com/AcademySoftwareFoundation/OpenTimelineIO
- **Purpose:** Editorial timeline interchange
- **Usage:** OTIO/EDL export for NLE integration

```
Copyright Contributors to the OpenTimelineIO project
Apache License 2.0
```

---

## Web Framework & UI

### Flask
- **License:** BSD 3-Clause License
- **Website:** https://flask.palletsprojects.com/
- **Purpose:** Web framework for UI
- **Usage:** Web interface backend

### Werkzeug
- **License:** BSD 3-Clause License
- **Website:** https://werkzeug.palletsprojects.com/
- **Purpose:** WSGI utilities
- **Usage:** Flask dependency

### Jinja2
- **License:** BSD 3-Clause License
- **Website:** https://jinja.palletsprojects.com/
- **Purpose:** Template engine
- **Usage:** HTML template rendering

---

## Machine Learning & Data

### NumPy
- **License:** BSD 3-Clause License
- **Website:** https://numpy.org/
- **Purpose:** Numerical computing
- **Usage:** Array operations, mathematical functions

### PyTorch
- **License:** BSD 3-Clause License
- **Website:** https://pytorch.org/
- **Purpose:** Deep learning framework
- **Usage:** AI model inference (Whisper, Demucs)

### torchaudio
- **License:** BSD 2-Clause License
- **Website:** https://pytorch.org/audio/
- **Purpose:** Audio processing with PyTorch
- **Usage:** Audio loading and processing

### transformers (Hugging Face)
- **License:** Apache License 2.0
- **Website:** https://huggingface.co/transformers
- **Purpose:** Pre-trained models
- **Usage:** Optional LLM integration

---

## Audio Processing

### soundfile
- **License:** BSD 3-Clause License
- **Website:** https://github.com/bastibe/python-soundfile
- **Purpose:** Audio file I/O
- **Usage:** Reading/writing audio files

### audioread
- **License:** MIT License
- **Website:** https://github.com/beetbox/audioread
- **Purpose:** Audio decoding
- **Usage:** Librosa dependency

### resampy
- **License:** ISC License
- **Website:** https://github.com/bmcfee/resampy
- **Purpose:** Audio resampling
- **Usage:** Librosa dependency

---

## Utilities

### psutil
- **License:** BSD 3-Clause License
- **Website:** https://github.com/giampaolo/psutil
- **Purpose:** System monitoring
- **Usage:** Memory/CPU monitoring

### tqdm
- **License:** MIT License / MPL 2.0
- **Website:** https://github.com/tqdm/tqdm
- **Purpose:** Progress bars
- **Usage:** CLI progress display

### pydantic
- **License:** MIT License
- **Website:** https://docs.pydantic.dev/
- **Purpose:** Data validation
- **Usage:** Configuration validation

### typer
- **License:** MIT License
- **Website:** https://typer.tiangolo.com/
- **Purpose:** CLI framework
- **Usage:** Command-line interface

---

## Fonts (UI)

### Space Grotesk
- **License:** SIL Open Font License 1.1
- **Website:** https://fonts.google.com/specimen/Space+Grotesk
- **Purpose:** UI typography
- **Usage:** Headlines, interface text

### Inter
- **License:** SIL Open Font License 1.1
- **Website:** https://rsms.me/inter/
- **Purpose:** UI typography
- **Usage:** Body text

### Space Mono
- **License:** SIL Open Font License 1.1
- **Website:** https://fonts.google.com/specimen/Space+Mono
- **Purpose:** Monospace typography
- **Usage:** Code, timestamps

---

## Full License Texts

### Apache License 2.0

```
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.
[Full license text available at: https://www.apache.org/licenses/LICENSE-2.0]
```

### MIT License

```
MIT License

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

### BSD 3-Clause License

```
BSD 3-Clause License

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

### ISC License

```
ISC License

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
```

### SIL Open Font License 1.1

```
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007

PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide development
of collaborative font projects, to support the font creation efforts of academic
and linguistic communities, and to provide a free and open framework in which
fonts may be shared and improved in partnership with others.

[Full license text available at: https://scripts.sil.org/OFL]
```

---

## Model Weights & Pre-trained Models

### Whisper Models
- **Source:** OpenAI
- **License:** MIT
- **Models Used:** tiny, base, small, medium
- **Note:** Model weights downloaded on first use

### HT Demucs Models
- **Source:** Meta AI Research
- **License:** MIT
- **Models Used:** htdemucs
- **Note:** Model weights downloaded on first use

### Real-ESRGAN Models
- **Source:** Xintao Wang et al.
- **License:** BSD 3-Clause
- **Models Used:** realesr-animevideov3
- **Note:** Model weights downloaded on first use

---

## Compliance Notes

### LGPL Compliance (FFmpeg)
- We dynamically link to FFmpeg libraries
- Source code for our modifications (if any) is available in this repository
- Users can replace FFmpeg with their own build
- We do not statically link GPL-only components

### Attribution Requirements
- This THIRD_PARTY_LICENSES.md file satisfies attribution requirements
- The NOTICE file in our distribution references this document
- Our documentation credits all major dependencies

### Model Usage
- All pre-trained models are used under their respective licenses
- We do not claim ownership of model weights
- Users downloading models agree to original license terms

---

## How to Update This Document

When adding new dependencies:

1. Identify the license type
2. Add entry to appropriate section
3. Include: name, license, website, purpose, usage
4. Add full license text if not already present
5. Update NOTICE file if attribution required

---

*Last updated: January 2026*
