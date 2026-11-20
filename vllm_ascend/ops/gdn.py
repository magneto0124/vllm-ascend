# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility import for GDN callers using the former module path."""

import sys

from vllm_ascend.attention import gdn

sys.modules[__name__] = gdn
