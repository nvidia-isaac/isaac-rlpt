# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Documentation site. `make html` renders _build/html/index.html; see README.md.

SPHINXBUILD  ?= sphinx-build
SPHINXOPTS   ?=
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html strict linkcheck serve clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# What CI publishes with: any warning (a dead link, a page missing from the navigation)
# fails the build instead of reaching the site.
strict:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going $(SPHINXOPTS)

# External links only, and never part of a normal build: it needs the network.
linkcheck:
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

serve: html
	@python3 -m http.server 8000 --bind 127.0.0.1 --directory "$(BUILDDIR)/html"

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
