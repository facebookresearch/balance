# Dependency compatibility patch

The website overrides `image-size` to the patched upstream 2.0.3 release. Docusaurus 2.4.3 still calls its removed file/callback API, so `patch-image-size.cjs` switches that one pinned consumer to `imageSizeFromFile`, the public Promise-based API. It runs during website postinstall; scripts-disabled installs must run it explicitly before building.

The patch checks the consumer version and exact import/call before changing the installed file, and is idempotent. Unexpected source or versions fail rather than silently dropping width/height metadata. It modifies only local `node_modules`, not upstream sources or installed package versions. `yarn test:image-size` covers real Docusaurus image metadata and bounds malformed ICNS/JXL/HEIF probes to three seconds.

Source advisories: [ICNS](https://github.com/advisories/GHSA-w3rx-r6r6-pgpr) and [JXL/HEIF](https://github.com/advisories/GHSA-5p2g-fcmc-qvqq).

Remove this compatibility patch when Docusaurus natively supports the fixed parser API, after keeping the image-dimension regression green. A successful site build alone is insufficient because the old transformer catches image-size errors.

## Local deepmerge mitigation

There is no published release fixing [GHSA-gcqx-w48m-w2h4](https://github.com/advisories/GHSA-gcqx-w48m-w2h4). The website pins 4.3.1 and `patch-deepmerge.cjs` rejects `__proto__`, `constructor`, and `prototype` keys in both target-copy and source-merge loops. It patches the source, CommonJS and UMD distributions, checking the version and exact loops before writing. This implements the key-name guard described in [upstream issue 273](https://github.com/TehShrike/deepmerge/issues/273), not an upstream released fix.

Those three reserved data keys are intentionally discarded, unlike some preceding edge cases. Ordinary nested options, arrays, symbols and supported merge options are tested; this is not a general-purpose sanitizer for custom callbacks or already-poisoned objects. `yarn test:deepmerge` checks the returned object's prototype, not merely global Object.prototype.

The package version stays 4.3.1, so vulnerability scanners may still report this advisory. Do not dismiss it as a released-version fix. Replace/remove the local mitigation when a verified patched release is adopted. Scripts-disabled installs must run both patch scripts explicitly; the website CI job runs the corresponding regressions after building.
