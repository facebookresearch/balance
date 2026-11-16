# Dependency compatibility patch

The website overrides `image-size` to the patched upstream 2.0.3 release. Docusaurus 2.4.3 still calls its removed file/callback API, so `patch-image-size.cjs` switches that one pinned consumer to `imageSizeFromFile`, the public Promise-based API. It runs during website postinstall; scripts-disabled installs must run it explicitly before building.

The patch checks the consumer version and exact import/call before changing the installed file, and is idempotent. Unexpected source or versions fail rather than silently dropping width/height metadata. It modifies only local `node_modules`, not upstream sources or installed package versions. `yarn test:image-size` covers real Docusaurus image metadata and bounds malformed ICNS/JXL/HEIF probes to three seconds.

Source advisories: [ICNS](https://github.com/advisories/GHSA-w3rx-r6r6-pgpr) and [JXL/HEIF](https://github.com/advisories/GHSA-5p2g-fcmc-qvqq).

Remove this compatibility patch when Docusaurus natively supports the fixed parser API, after keeping the image-dimension regression green. A successful site build alone is insufficient because the old transformer catches image-size errors.
