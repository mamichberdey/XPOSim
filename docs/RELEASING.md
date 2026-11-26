# Publishing XPOSim releases

Source code lives on GitHub. `pip install XPOSim` downloads a versioned wheel
from PyPI after the first upload. Pushing ordinary commits to `main` runs CI;
it does not publish an unversioned package or update installed copies.
Pushing a `v*` tag starts `.github/workflows/publish.yml` automatically.

## One-time PyPI account setup

The repository workflow is ready for PyPI's GitHub Trusted Publishing. This
binding must be created by the owner of a PyPI account; it cannot be configured
by committing YAML to GitHub. Use your own account and keep its credentials
private. PyPI requires two-factor authentication for publishing accounts.

1. Sign in to [PyPI](https://pypi.org/account/login/).
2. For the first upload, open
   [Publishing / add a pending publisher](https://pypi.org/manage/account/publishing/)
   and select GitHub. If the project already belongs to your account, add the
   same publisher under the project's Publishing settings instead.
3. Enter these exact values:

   | PyPI form field | Value |
   | --- | --- |
   | PyPI project name (pending publisher only) | `XPOSim` |
   | Owner | `mamichberdey` |
   | Repository name | `XPOSim` |
   | Workflow name / filename | `publish.yml` |
   | Environment name | `pypi` |

4. The workflow uses the GitHub environment `pypi`. GitHub creates it automatically
   on first use if it does not exist. If you configure it in
   [repository settings](https://github.com/mamichberdey/XPOSim/settings/environments),
   allow the release tags. Required reviewers are optional; leave that gate off
   for unattended publication after the tests pass.
5. Then push the first matching tag as described below.

No `PYPI_API_TOKEN` or PyPI password is stored in GitHub. The publish job obtains
an OIDC identity with `id-token: write`, bound to the repository, workflow and
`pypi` environment. The build job has read-only repository permissions. The
publish job downloads the already tested artifact and does not check out or
execute the project source. PyPI's publishing action creates provenance
attestations using its Trusted Publishing support.

The PyPI project JSON endpoint returned 404 during preparation on 26 November
2026. This is not a reservation of the project name; PyPI determines whether
it can be created when the pending publisher is used.

## First release

Once the PyPI publisher and GitHub environment are configured and these files
are on `main`, the current version can be released:

```sh
git switch main
git pull --ff-only
git tag -a v0.38.0 -m "XPOSim 0.38.0"
git push origin v0.38.0
```

An authenticated GitHub maintainer can also create a Release targeting `main`
with a new tag `v0.38.0`; creating the tag triggers the same workflow. A draft
release without a pushed tag does not publish a package. After a tag has been
pushed, the workflow publishes even if a GitHub Release is still a draft.

Monitor [Publish to PyPI](https://github.com/mamichberdey/XPOSim/actions/workflows/publish.yml).
The workflow checks that the tag commit is part of `main`, that its tag matches
`XPOSim.__version__`, and that license/provenance files are present. It then runs
regressions, builds the wheel and sdist, checks PyPI metadata with Twine and
verifies offline installation, unchanged runtime dependencies, bundled data and
license files. Only then does the separate publish job upload those exact files.

## Later releases

1. Merge tested changes into `main`.
2. Change `__version__` in `XPOSim.py` to a new version, for example `0.39.0`.
3. Update release notes and version-specific examples. If runtime code changes,
   refresh the Colab notebook's source commit and SHA256 after committing the
   module. The source pin must identify the actual module bytes.
4. Push the version change to `main`, then create and push `v0.39.0`.
5. Confirm the workflow and the [PyPI project page](https://pypi.org/project/XPOSim/).

Every published version is immutable. Do not reuse or move a published release
tag. To correct a release, increment the version and publish a new tag. If
publication fails before any upload because the PyPI binding is missing, add the
correct binding and rerun that failed workflow. A partially uploaded version
requires inspecting PyPI and generally issuing a new version; the workflow does
not silently skip existing files or rebuild in its publish job.

Users update explicitly:

```sh
python -m pip install --upgrade XPOSim
```

XPOSim does not update itself inside a running notebook or modify installed
numerical frameworks. Restart a Python/Jupyter process after upgrading. GPU
framework selection and driver compatibility remain environment responsibilities.

## Licensing and citation

The maintainers selected MIT for XPOSim code. Material tables are separately
attributed to DABAX and retain its MIT copyright/permission notice, their
original headers and a pinned provenance record. See `LICENSE`,
`THIRD_PARTY_NOTICES.md` and `xray_material_data/README.md`. The distribution
checker verifies that these notices are present and unchanged in the artifacts.

`CITATION.cff` is intentionally deferred at the author's request. It is not a
PyPI requirement. Later, a software citation can be added independently of a
paper; a published paper can be supplied as `preferred-citation` when available.
Do not invent a DOI, publication date or author identity.

## Local release checks

Use an isolated development environment with NumPy and the build tools. Runtime
package metadata remains `dependencies = []`.

```sh
python -m pip install 'setuptools>=77.0.3' wheel numpy build twine
python tools/check_release.py v0.38.0
python -m build
python -m twine check --strict dist/*
python -m pip download --only-binary=:all: --no-deps numpy --dest build/runtime-wheels
python tests/check_distribution.py dist build/runtime-wheels
```

Run numerical tests with the development tools described in `docs/VALIDATION.md`.
Official references:
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/),
[pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/),
[PyPA GitHub Actions publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/).
